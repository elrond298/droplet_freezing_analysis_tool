from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from tube_detection import detect_inner_circles, infer_missing_tubes, locate_pcr_tubes


DEFAULT_IMAGE = REPO_ROOT / "1/data/images/2023-04-03_16-07-57.png"
DEFAULT_LOCATIONS = REPO_ROOT / "1/inner_circles_20260310_173939.pkl"


def parse_tubes_size(value: str) -> tuple[int, int]:
	parts = [part.strip() for part in value.split(",")]
	if len(parts) != 2:
		raise argparse.ArgumentTypeError("tubes size must be in the form rows,cols")

	try:
		rows, cols = (int(part) for part in parts)
	except ValueError as error:
		raise argparse.ArgumentTypeError("tubes size must contain integers") from error

	return rows, cols


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Visualize detected inner-circle locations on an image.",
	)
	parser.add_argument(
		"image",
		nargs="?",
		default=str(DEFAULT_IMAGE),
		help="Path to the source image.",
	)
	parser.add_argument(
		"--locations",
		default=str(DEFAULT_LOCATIONS),
		help="Optional pickle/json file containing saved inner-circle locations.",
	)
	parser.add_argument(
		"--detect",
		action="store_true",
		help="Ignore --locations and detect inner circles directly from the image.",
	)
	parser.add_argument(
		"--output",
		help="Output path for the annotated image. Defaults beside the source image.",
	)
	parser.add_argument(
		"--min-area",
		type=int,
		default=100,
		help="Minimum contour area for PCR tube detection.",
	)
	parser.add_argument(
		"--circularity-threshold",
		type=float,
		default=0.2,
		help="Circularity threshold for PCR tube detection.",
	)
	parser.add_argument(
		"--tubes-size",
		type=parse_tubes_size,
		default=(16, 10),
		help="Expected tube grid size as rows,cols.",
	)
	parser.add_argument(
		"--roi-size",
		type=int,
		default=30,
		help="ROI size used when detecting inner circles.",
	)
	parser.add_argument(
		"--radius",
		type=int,
		default=10,
		help="Inner-circle radius used for visualization and detection fallback.",
	)
	return parser.parse_args()


def load_circle_locations(path: Path) -> list[dict]:
	if not path.exists():
		raise FileNotFoundError(f"Location file not found: {path}")

	if path.suffix.lower() == ".json":
		with path.open("r", encoding="utf-8") as handle:
			circles = json.load(handle)
	else:
		with path.open("rb") as handle:
			circles = pickle.load(handle)

	if not isinstance(circles, list):
		raise ValueError(f"Expected a list of circles in {path}")

	normalized = []
	for circle in circles:
		normalized.append(
			{
				"x": int(round(circle["x"])),
				"y": int(round(circle["y"])),
				"radius": int(round(circle.get("radius", 10))),
				"method": circle.get("method", "loaded"),
			}
		)
	return normalized


def detect_circles_from_image(image, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
	pcr_tubes, _ = locate_pcr_tubes(
		image,
		min_area=args.min_area,
		circularity_threshold=args.circularity_threshold,
	)
	inferred_tubes = infer_missing_tubes(
		pcr_tubes,
		image.shape,
		tubes_size=args.tubes_size,
		rotate="auto",
	)
	all_tubes = pcr_tubes + inferred_tubes
	inner_circles = detect_inner_circles(
		image,
		all_tubes,
		roi_size=args.roi_size,
		radius=args.radius,
	)
	return all_tubes, inner_circles


def annotate_image(image, tubes: list[dict], inner_circles: list[dict]):
	annotated = image.copy()

	for tube in tubes:
		color = (0, 255, 0) if "inferred" not in tube else (0, 0, 255)
		cv2.circle(annotated, (int(tube["x"]), int(tube["y"])), int(tube["radius"]), color, 2)

	for circle in inner_circles:
		center = (int(circle["x"]), int(circle["y"]))
		radius = int(circle.get("radius", 10))
		method = circle.get("method", "loaded")
		color = (255, 0, 255) if method == "loaded" else (0, 0, 0)
		cv2.circle(annotated, center, radius, color, 1)
		cv2.drawMarker(
			annotated,
			center,
			color,
			markerType=cv2.MARKER_CROSS,
			markerSize=max(8, radius * 2),
			thickness=1,
		)

	status = f"Tubes: {len(tubes)} | Inner circles: {len(inner_circles)}"
	cv2.putText(
		annotated,
		status,
		(16, 32),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.8,
		(255, 255, 255),
		2,
		cv2.LINE_AA,
	)
	cv2.putText(
		annotated,
		status,
		(16, 32),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.8,
		(0, 128, 255),
		1,
		cv2.LINE_AA,
	)
	return annotated, status


def main() -> None:
	args = parse_args()
	image_path = Path(args.image)
	output_path = Path(args.output) if args.output else image_path.with_name(f"{image_path.stem}_inner_circles.png")

	image = cv2.imread(str(image_path))
	if image is None:
		raise FileNotFoundError(f"Unable to read image: {image_path}")

	if args.detect:
		tubes, inner_circles = detect_circles_from_image(image, args)
		source = "detected from image"
	else:
		inner_circles = load_circle_locations(Path(args.locations))
		tubes = []
		source = f"loaded from {args.locations}"

	annotated, status = annotate_image(image, tubes, inner_circles)
	if not cv2.imwrite(str(output_path), annotated):
		raise RuntimeError(f"Failed to write annotated image: {output_path}")

	print(status)
	print(f"Inner-circle source: {source}")
	print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
	main()