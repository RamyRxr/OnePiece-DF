// Reference Unity client for the stretch perception service.
//
// Copy this into the Unity project (Assets/Scripts/Vision/) and add
// "com.unity.nuget.newtonsoft-json" via the Package Manager.
//
// The service streams a small JSON state object at 30Hz over a local
// websocket. The socket runs on a background thread, but it only queues the
// raw messages: they are parsed, stored and turned into events on Unity's
// main thread, in Update(). So every event handler below may touch
// GameObjects, transforms and Time freely.
//
//   var client = gameObject.AddComponent<StretchSignalClient>();
//   client.OnRelease += (channel, peak) => { ... };
//
//   // each frame
//   float stretch = client.GetMagnitude("face.pullL");
//
// Values are DIMENSIONLESS. They express "how much more than your neutral
// pose", scaled by a body-derived reference length. Multiply by a per-channel
// gain to get centimetres of avatar stretch - that gain is a game-feel
// decision and belongs in Unity, not in the vision service.
//
// The main interaction is PINCH-GRAB-PULL. The player pinches an ear, the
// nose, the jaw, a cheek, the mouth, or any finger of their other hand,
// pulls their hand away, and lets go. Each hand grabs on its own - one hand,
// or both at once - so every grab channel names the pinching hand, <hand>
// being Left or Right:
//   grab.<hand>.active        1 while that hand holds a grab
//   grab.<hand>.stretch       how far it has pulled, in interocular distances
//                             (palm lengths when a finger is grabbed)
//   grab.<hand>.stretchX / Y  the pull vector, same units (+x = player's right, +y = down)
//   grab.<hand>.target        1 cheekL 2 cheekR 3 earL 4 earR 5 nose 6 chin 7 jawL 8 jawR
//                             9 mouthL 10 mouthR 11 lipTop 12 lipBottom
//                             20-24 left thumb..pinky, 25-29 right thumb..pinky
//   OnGrabStretch             raised every frame for each hand holding a grab
//   OnRelease("grab.Left.stretch", peak)   a let-go: fire the Gum-Gum Pistol here
//
// A pull is UNSIGNED. Tugging your cheek down shrinks one measurement while
// tugging it out grows another, but the player stretched their face either
// way, so the service reports the magnitude of the deviation. GetMagnitude()
// is what you normally want; GetValue() is the signed value, for the rare
// case where the direction matters.
//
// The face channels you will want:
//   face.pullL / face.pullR  - how hard that side is being pulled. These are
//                              the only face channels that fire for a tug in
//                              ANY direction, so bind the avatar's cheek and
//                              ear to these.
//   face.pullAsym            - the difference between the two sides. Large
//                              when only one side is being pulled.
//   face.cheekL / cheekR     - more directional views of the same motion,
//                              useful for localising the pull.
//   face.earL / earR         - the same, biased towards an outward tug.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;

namespace OnePieceDF.Vision
{
    public class StretchSignalClient : MonoBehaviour
    {
        /// <summary>The wire format version this client understands.</summary>
        public const int ProtocolVersion = 3;

        /// <summary>The hands that grab, as named in the grab channels.</summary>
        public static readonly string[] GrabHands = { "Left", "Right" };

        [Tooltip("Host of the vision service. Localhost unless the service runs on another machine.")]
        [SerializeField] private string host = "127.0.0.1";

        [SerializeField] private int port = 8765;

        [Tooltip("Reconnect delay in seconds if the service is not running.")]
        [SerializeField] private float reconnectDelay = 2f;

        /// <summary>Raised when a pull is released. channel is e.g. "face.cheekL",
        /// or "grab.Left.stretch" for a pinch-grab. peak is the magnitude of the pull -
        /// always positive.</summary>
        public event Action<string, float> OnRelease;

        /// <summary>Raised when a pull is released, with the direction it went in.</summary>
        public event Action<string, float, int> OnReleaseWithDirection;

        /// <summary>Raised when the service connects or disconnects.</summary>
        public event Action<bool> OnConnectionChanged;

        /// <summary>Raised every frame for each hand holding a pinch-grab:
        /// (hand, target, stretch, stretchX, stretchY). See the header for target codes.</summary>
        public event Action<string, int, float, float, float> OnGrabStretch;

        // Messages queued while Update() is not running (component disabled, a
        // long scene load, an unfocused build) beyond this are dropped: about
        // two seconds at 30Hz. Their state is stale and their releases are
        // shots the player fired long ago.
        private const int MaxQueuedMessages = 60;

        // Only the newest few messages of a backlog fire their events.
        private const int MaxEventMessagesPerFrame = 3;

        private enum InboundKind { Message, Connected, Disconnected }

        private readonly struct Inbound
        {
            public readonly InboundKind Kind;
            public readonly string Payload;

            public Inbound(InboundKind kind, string payload = null)
            {
                Kind = kind;
                Payload = payload;
            }
        }

        // One queue, so a disconnect can never overtake the last messages.
        // Filled by the socket thread, drained by Update() on the main thread.
        private readonly ConcurrentQueue<Inbound> _inbox = new ConcurrentQueue<Inbound>();
        private int _queuedMessages;
        private readonly List<Inbound> _drained = new List<Inbound>();

        // Main thread only.
        private readonly Dictionary<string, float> _values = new Dictionary<string, float>();
        private readonly Dictionary<string, float> _magnitudes = new Dictionary<string, float>();
        private readonly Dictionary<string, bool> _active = new Dictionary<string, bool>();

        private ClientWebSocket _socket;
        private CancellationTokenSource _cts;
        private bool _running;
        private float _nextReconnectAt;
        private bool _warnedAboutVersion;

        public bool IsConnected { get; private set; }

        /// <summary>True when the player is present and the face is being tracked.</summary>
        public bool FaceTracked { get; private set; }

        /// <summary>Latest signed value for a channel, or 0 if the channel is missing.</summary>
        public float GetValue(string channel)
        {
            return _values.TryGetValue(channel, out float value) ? value : 0f;
        }

        /// <summary>Latest magnitude of the deviation from neutral, or 0 if missing.
        /// This is the number to drive the rig with - a pull is unsigned.</summary>
        public float GetMagnitude(string channel)
        {
            return _magnitudes.TryGetValue(channel, out float value) ? value : 0f;
        }

        /// <summary>True when the channel is currently pulled past its threshold.</summary>
        public bool IsActive(string channel)
        {
            return _active.TryGetValue(channel, out bool active) && active;
        }

        private void Start()
        {
            Connect();
        }

        private void Update()
        {
            DrainInbox();

            // Unscaled, so a paused game (timeScale 0) still reconnects.
            if (!_running && Time.unscaledTime >= _nextReconnectAt)
            {
                Connect();
            }
        }

        private void OnDestroy()
        {
            Disconnect();
        }

        // --- connection (main thread) ------------------------------------------

        private void Connect()
        {
            Disconnect();
            _running = true;
            _cts = new CancellationTokenSource();
            _socket = new ClientWebSocket();

            // The task works on its own references, so a later Connect() that
            // replaces the fields cannot pull the socket out from under it.
            ClientWebSocket socket = _socket;
            CancellationToken token = _cts.Token;
            var uri = new Uri($"ws://{host}:{port}");
            _ = Task.Run(() => RunConnection(socket, uri, token));
        }

        private void Disconnect()
        {
            _running = false;
            _cts?.Cancel();
            try
            {
                _socket?.Abort();
            }
            catch (Exception)
            {
                // Already closed.
            }
            _socket?.Dispose();
            _socket = null;
            _cts?.Dispose();
            _cts = null;
        }

        private void DrainInbox()
        {
            _drained.Clear();
            int messages = 0;
            while (_inbox.TryDequeue(out Inbound item))
            {
                _drained.Add(item);
                if (item.Kind == InboundKind.Message)
                {
                    messages++;
                    Interlocked.Decrement(ref _queuedMessages);
                }
            }

            // In order. Every message updates the channel values; only the
            // newest few of a backlog raise events.
            int seen = 0;
            foreach (Inbound item in _drained)
            {
                if (item.Kind == InboundKind.Message)
                {
                    seen++;
                    HandleMessage(item.Payload, seen > messages - MaxEventMessagesPerFrame);
                }
                else
                {
                    HandleConnectionChange(item.Kind == InboundKind.Connected);
                }
            }
        }

        private void HandleConnectionChange(bool connected)
        {
            IsConnected = connected;
            if (connected)
            {
                Debug.Log($"[Vision] connected to ws://{host}:{port}");
            }
            else
            {
                FaceTracked = false;
                _running = false;
                _nextReconnectAt = Time.unscaledTime + reconnectDelay;
            }
            OnConnectionChanged?.Invoke(connected);
        }

        // --- socket (background thread) ----------------------------------------
        //
        // Nothing in here may touch Unity objects, Time, or the public events.
        // Debug.Log is the one Unity call that is safe from any thread.

        private async Task RunConnection(ClientWebSocket socket, Uri uri, CancellationToken token)
        {
            try
            {
                await socket.ConnectAsync(uri, token);
                _inbox.Enqueue(new Inbound(InboundKind.Connected));
                await ReceiveLoop(socket, token);
            }
            catch (OperationCanceledException)
            {
                // Shutting down.
            }
            catch (ObjectDisposedException)
            {
                // Disconnect() disposed the socket mid-call.
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"[Vision] connection failed: {exception.Message}");
            }
            finally
            {
                _inbox.Enqueue(new Inbound(InboundKind.Disconnected));
            }
        }

        private async Task ReceiveLoop(ClientWebSocket socket, CancellationToken token)
        {
            var buffer = new byte[16 * 1024];
            var builder = new StringBuilder();

            while (socket.State == WebSocketState.Open && !token.IsCancellationRequested)
            {
                WebSocketReceiveResult result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), token);
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    break;
                }

                builder.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                if (!result.EndOfMessage)
                {
                    continue;
                }

                // Over the cap, drop the newest: Update() is not running, and
                // when it resumes the next live message replaces it anyway.
                if (Volatile.Read(ref _queuedMessages) < MaxQueuedMessages)
                {
                    Interlocked.Increment(ref _queuedMessages);
                    _inbox.Enqueue(new Inbound(InboundKind.Message, builder.ToString()));
                }
                builder.Clear();
            }
        }

        // --- messages (main thread) --------------------------------------------

        private void HandleMessage(string payload, bool raiseEvents)
        {
            StateMessage message;
            try
            {
                message = JsonConvert.DeserializeObject<StateMessage>(payload);
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"[Vision] bad payload: {exception.Message}");
                return;
            }

            if (message?.Channels == null)
            {
                return;
            }

            if (message.Version != ProtocolVersion && !_warnedAboutVersion)
            {
                _warnedAboutVersion = true;
                Debug.LogWarning($"[Vision] service speaks protocol v{message.Version}, this client expects v{ProtocolVersion}");
            }

            FaceTracked = message.Quality != null && message.Quality.Face > 0;

            foreach (KeyValuePair<string, ChannelState> entry in message.Channels)
            {
                _values[entry.Key] = entry.Value.Value;
                _magnitudes[entry.Key] = entry.Value.Magnitude;
                _active[entry.Key] = entry.Value.Active > 0;
            }

            if (!raiseEvents)
            {
                return;
            }

            foreach (string hand in GrabHands)
            {
                string grab = "grab." + hand;
                if (GetValue(grab + ".active") > 0.5f)
                {
                    OnGrabStretch?.Invoke(
                        hand,
                        Mathf.RoundToInt(GetValue(grab + ".target")),
                        GetValue(grab + ".stretch"),
                        GetValue(grab + ".stretchX"),
                        GetValue(grab + ".stretchY"));
                }
            }

            if (message.Events == null)
            {
                return;
            }

            foreach (ReleaseEvent release in message.Events)
            {
                OnRelease?.Invoke(release.Channel, release.Peak);
                OnReleaseWithDirection?.Invoke(release.Channel, release.Peak, release.Direction);
            }
        }

        // --- wire format -----------------------------------------------------

        [Serializable]
        private class StateMessage
        {
            [JsonProperty("v")] public int Version;
            [JsonProperty("t")] public long TimestampMs;
            [JsonProperty("ch")] public Dictionary<string, ChannelState> Channels;
            [JsonProperty("ev")] public ReleaseEvent[] Events;
            [JsonProperty("q")] public QualityState Quality;
        }

        [Serializable]
        private class ChannelState
        {
            [JsonProperty("v")] public float Value;
            [JsonProperty("m")] public float Magnitude;
            [JsonProperty("a")] public int Active;
            [JsonProperty("l")] public int Lost;
        }

        [Serializable]
        private class ReleaseEvent
        {
            [JsonProperty("ch")] public string Channel;
            [JsonProperty("peak")] public float Peak;
            [JsonProperty("d")] public int Direction;
            [JsonProperty("v")] public float Value;
            [JsonProperty("t")] public long TimestampMs;
        }

        [Serializable]
        private class QualityState
        {
            [JsonProperty("face")] public int Face;
            [JsonProperty("handL")] public int HandLeft;
            [JsonProperty("handR")] public int HandRight;
            [JsonProperty("pose")] public int Pose;
            [JsonProperty("fps")] public float Fps;
            [JsonProperty("cal")] public int Calibrated;
        }
    }
}
