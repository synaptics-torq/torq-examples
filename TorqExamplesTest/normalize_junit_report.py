import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SUITE_NAME = "TorqExamplesTest"

DEMO_CLASS_NAMES = {
    "gemma3": "Gemma3",
    "moonshine": "Moonshine",
    "moonshine_streaming": "MoonshineStreaming",
    "object_detection": "ObjectDetection",
    "pose_estimation": "PoseEstimation",
    "LiquidAI-LFM2.5-230M": "LiquidAILFM25230M",
    "LiquidAI-LFM2-VL-450M": "LiquidAILFM2VL450M",
}

DEMO_MODEL_NAMES = {
    "gemma3": "gemma-3-270m-it-torq",
    "moonshine": "moonshine-tiny-bf16-torq",
    "moonshine_streaming": "moonshine-streaming-tiny-torq",
    "object_detection": "yolov8-od-nano-320-int8-torq",
    "pose_estimation": "yolov8-pose-nano-320-int8-torq",
    "LiquidAI-LFM2.5-230M": "LiquidAI-LFM2.5-230M",
    "LiquidAI-LFM2-VL-450M": "LiquidAI-LFM2-VL-450M",
}


def _demo_display_name(demo_name: str) -> str:
    if demo_name in DEMO_CLASS_NAMES:
        return DEMO_CLASS_NAMES[demo_name]
    parts = re.split(r"[^A-Za-z0-9]+", demo_name)
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


def _demo_model_name(demo_name: str) -> str:
    return DEMO_MODEL_NAMES.get(demo_name, demo_name)


def _extract_demo_name(testcase_name: str) -> str | None:
    match = re.search(r"\[(.+)\]$", testcase_name)
    if match:
        return match.group(1)
    return None


def _normalize_testcase(testcase: ET.Element) -> None:
    name = testcase.attrib.get("name", "")
    demo_name = _extract_demo_name(name)
    if not demo_name:
        return

    demo_display = _demo_display_name(demo_name)
    model_name = _demo_model_name(demo_name)

    if name.startswith("test_demo_setup_on_dut["):
        testcase.set("classname", f"{SUITE_NAME}.TestSetup")
        testcase.set("name", f"TestSetup.test_setup{demo_display}")
        return

    if name.startswith("test_demo_inference_on_dut["):
        testcase.set("classname", f"{SUITE_NAME}.TestDemo{demo_display}")
        testcase.set("name", f"TestDemo{demo_display}.test_{model_name}")


def normalize_junit_report(report_path: Path) -> None:
    tree = ET.parse(report_path)
    root = tree.getroot()

    if root.tag == "testsuites":
        suite = root.find("testsuite")
        if suite is None:
            raise RuntimeError("No <testsuite> element found under <testsuites>")
    elif root.tag == "testsuite":
        suite = root
    else:
        raise RuntimeError(f"Unsupported JUnit root element: {root.tag}")

    suite.set("name", SUITE_NAME)

    for testcase in suite.findall("testcase"):
        _normalize_testcase(testcase)

    if root.tag == "testsuites":
        new_root = ET.Element("testsuite", suite.attrib)
        for child in list(suite):
            new_root.append(child)
        tree._setroot(new_root)

    tree.write(report_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python normalize_junit_report.py <report.xml>")
    normalize_junit_report(Path(sys.argv[1]))
