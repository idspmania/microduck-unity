using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.UI;

namespace Microduck
{
    public class MicroduckView : MonoBehaviour
    {
        public MicroduckSimulation Simulation;
        public Camera ViewCamera;
        public TMP_FontAsset Font;
        TMP_Text stats;
        float yaw = -55, elevation = 22, distance = .85f;
        readonly Color ink = new Color(.81f, .88f, .9f);

        void Start()
        {
            if (!FindAnyObjectByType<EventSystem>())
                new GameObject("EventSystem", typeof(EventSystem), typeof(InputSystemUIInputModule));
            var canvasGo = new GameObject("Microduck Controls", typeof(Canvas), typeof(UnityEngine.UI.CanvasScaler), typeof(UnityEngine.UI.GraphicRaycaster));
            canvasGo.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGo.GetComponent<UnityEngine.UI.CanvasScaler>();
            scaler.uiScaleMode = UnityEngine.UI.CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1440, 900); scaler.matchWidthOrHeight = .5f;
            var header = Panel(canvasGo.transform, "Header", new Vector2(0, 1), new Vector2(28, -24), new Vector2(380, 204));
            Label(header, "MICRODUCK / POLICY LAB", new Vector2(18, -14), new Vector2(350, 24), 15, new Color(.7f, .95f, .3f));
            Label(header, "Walking · model 1500", new Vector2(18, -42), new Vector2(350, 32), 25, Color.white);
            stats = Label(header, "Starting simulation…", new Vector2(18, -86), new Vector2(350, 112), 15, ink);
            var controls = Panel(canvasGo.transform, "Controls", new Vector2(0, 0), new Vector2(28, 170), new Vector2(650, 142));
            Label(controls, "W/S  forward/back    A/D  strafe    Q/E  turn", new Vector2(18, -12), new Vector2(610, 24), 16, ink);
            Label(controls, "Space  idle    R  reset    P  pause    Right drag / scroll  camera", new Vector2(18, -38), new Vector2(620, 24), 14, ink);
            Button(controls, "Walk", 18, () => SetCommand(new Vector3(.2f, 0, 0)));
            Button(controls, "Idle", 118, () => SetCommand(Vector3.zero));
            Button(controls, "Turn L", 218, () => SetCommand(new Vector3(0, 0, .6f)));
            Button(controls, "Turn R", 318, () => SetCommand(new Vector3(0, 0, -.6f)));
            Button(controls, "Reset", 418, () => Simulation.ResetRobot());
            Button(controls, "Pause", 518, () => Simulation.Paused = !Simulation.Paused);
        }
        void SetCommand(Vector3 command) { Simulation.VelocityCommand = command; Simulation.Paused = false; }
        RectTransform Panel(Transform parent, string name, Vector2 anchor, Vector2 position, Vector2 size)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(UnityEngine.UI.Image)); go.transform.SetParent(parent, false);
            var rt = go.GetComponent<RectTransform>(); rt.anchorMin = rt.anchorMax = anchor; rt.pivot = new Vector2(0, 1); rt.anchoredPosition = position; rt.sizeDelta = size;
            var image = go.GetComponent<UnityEngine.UI.Image>(); image.color = new Color(.035f, .065f, .08f, .93f); image.raycastTarget = false;
            return rt;
        }
        TMP_Text Label(Transform parent, string text, Vector2 position, Vector2 size, int fontSize, Color color)
        {
            var go = new GameObject("Text", typeof(RectTransform), typeof(TextMeshProUGUI)); go.transform.SetParent(parent, false);
            var rt = go.GetComponent<RectTransform>(); rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0, 1); rt.anchoredPosition = position; rt.sizeDelta = size;
            var t = go.GetComponent<TextMeshProUGUI>(); t.font = Font; t.text = text; t.fontSize = fontSize; t.color = color; t.raycastTarget = false;
            return t;
        }
        void Button(Transform parent, string title, float x, UnityEngine.Events.UnityAction action)
        {
            var panel = Panel(parent, title, new Vector2(0, 1), new Vector2(x, -82), new Vector2(90, 38));
            var image = panel.GetComponent<UnityEngine.UI.Image>(); image.color = new Color(.19f, .28f, .3f); image.raycastTarget = true;
            var button = panel.gameObject.AddComponent<UnityEngine.UI.Button>(); button.targetGraphic = image; button.onClick.AddListener(action);
            var label = Label(panel, title, new Vector2(0, 0), new Vector2(90, 38), 16, Color.white); label.alignment = TextAlignmentOptions.Center;
        }
        void LateUpdate()
        {
            if (!Simulation || !Simulation.Ready) return;
            var mouse = Mouse.current;
            if (mouse != null)
            {
                if (mouse.rightButton.isPressed) { var delta = mouse.delta.ReadValue(); yaw += delta.x * .2f; elevation = Mathf.Clamp(elevation - delta.y * .15f, 5, 75); }
                distance = Mathf.Clamp(distance - mouse.scroll.ReadValue().y * .001f, .35f, 3);
            }
            Vector3 target = Simulation.BasePosition + Vector3.up * .05f;
            ViewCamera.transform.position = target + Quaternion.Euler(elevation, yaw, 0) * new Vector3(0, 0, -distance);
            ViewCamera.transform.LookAt(target);
            if (stats)
            {
                var v = Simulation.VelocityCommand;
                stats.text = $"{(Simulation.Paused ? "PAUSED" : "LIVE")}    {Simulation.SimulationTime:F1} s    steps {Simulation.PolicySteps}\n" +
                    $"Command   {v.x:+0.00;-0.00;0.00} / {v.y:+0.00;-0.00;0.00} m/s   {v.z:+0.0;-0.0;0.0} rad/s\n" +
                    $"Tilt {Simulation.TiltDegrees:F1}°    resets {Simulation.Falls}\n" +
                    $"Sentis CPU  {Simulation.LastInferenceMilliseconds:F2} ms\n" +
                    $"BAM M6  {Simulation.EffectiveVoltage:F2} V    50 / 200 Hz";
            }
        }
    }
}
