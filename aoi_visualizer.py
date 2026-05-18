import cv2
import matplotlib.pyplot as plt
import os

# IMAGE PATH
image_path = "Capstone.yolov8/train/images"

# LABEL PATH
label_path = "Capstone.yolov8/train/labels"

# CLASS NAMES
classes = {
    0: "CTA",
    1: "Headline",
    2: "Price",
    3: "Product",
    4: "logo"
}

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Global state is kept minimal so onclick can consume the current poster session.
aoi_boxes = []
attention_scores = {}


def list_available_posters(folder_path):
    """Return sorted poster filenames filtered by supported image extensions."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Image folder not found: {folder_path}")

    posters = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    return sorted(posters)


def load_poster(folder_path):
    """
    Dynamically select any poster either by index or filename.
    This supports future per-poster gaze pipelines without hardcoding names.
    """
    posters = list_available_posters(folder_path)

    if not posters:
        raise FileNotFoundError("No supported poster images found in the image folder.")

    print("Available Posters:")
    for idx, poster_name in enumerate(posters, start=1):
        print(f"{idx}. {poster_name}")

    lower_to_original = {name.lower(): name for name in posters}

    while True:
        selection = input("\nSelect poster by number OR type filename: ").strip()

        if not selection:
            print("Invalid input. Enter a number or a valid filename.")
            continue

        if selection.isdigit():
            selected_index = int(selection)
            if 1 <= selected_index <= len(posters):
                selected_poster = posters[selected_index - 1]
                break

            print(f"Invalid number. Choose between 1 and {len(posters)}.")
            continue

        manual_name = selection.lower()
        if manual_name in lower_to_original:
            selected_poster = lower_to_original[manual_name]
            break

        print("Filename not found. Please choose from the listed posters.")

    poster_full_path = os.path.join(folder_path, selected_poster)
    image = cv2.imread(poster_full_path)

    if image is None:
        raise ValueError(f"Unable to read selected poster: {selected_poster}")

    print(f"Selected Poster: {selected_poster}")
    return selected_poster, image


def load_labels(folder_path, poster_name):
    """
    Load label file that matches the selected poster base name.
    Example: burger.jpg -> burger.txt, burger.png -> burger.txt.
    """
    poster_stem, _ = os.path.splitext(poster_name)
    label_file = f"{poster_stem}.txt"
    label_full_path = os.path.join(folder_path, label_file)

    if not os.path.isfile(label_full_path):
        raise FileNotFoundError(f"Matching label file not found: {label_file}")

    with open(label_full_path, "r") as f:
        lines = f.readlines()

    print(f"Loaded Label File: {label_file}")
    return lines


def draw_aoi_boxes(image, lines, class_names):
    """Draw AOI boxes and labels for the currently selected poster."""
    h, w, _ = image.shape
    poster_aoi_boxes = []

    for line in lines:
        data = line.strip().split()
        if len(data) < 5:
            continue

        class_id = int(data[0])

        x_center = float(data[1]) * w
        y_center = float(data[2]) * h
        box_width = float(data[3]) * w
        box_height = float(data[4]) * h

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = class_names.get(class_id, f"class_{class_id}")
        poster_aoi_boxes.append((label, x1, y1, x2, y2))

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, poster_aoi_boxes


def onclick(event):
    """
    Keep click-based gaze simulation logic intact for now.
    This is where Tobii gaze-stream events can later be routed per poster session.
    """
    if event.xdata is None or event.ydata is None:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    print(f"\nGaze Point: ({x}, {y})")

    found = False
    fixation_time = 300

    for box in aoi_boxes:
        label, x1, y1, x2, y2 = box

        if x1 <= x <= x2 and y1 <= y <= y2:
            print(f"Looked at AOI: {label}")
            if label not in attention_scores:
                attention_scores[label] = 0
            attention_scores[label] += fixation_time
            print(f"Total Attention on {label}: {attention_scores[label]} ms")
            total_attention = sum(attention_scores.values())
            print(f"Overall Attention: {total_attention} ms")
            attention_percent = (attention_scores[label] / total_attention) * 100

            print(
                f"{label} Attention Percentage: "
                f"{attention_percent:.2f}%"
            )

            print("\nAOI Rankings:")

            sorted_aoi = sorted(
                attention_scores.items(),
                key=lambda item: item[1],
                reverse=True
            )

            rank = 1

            for aoi, score in sorted_aoi:
                percent = (score / total_attention) * 100

                print(
                    f"{rank}. {aoi} "
                    f"-> {percent:.2f}%"
                )

                rank += 1

            found = True

    if not found:
        print("Looked at Background")


def main():
    """Poster-session entrypoint. One run = one poster analytics session."""
    global aoi_boxes
    global attention_scores

    selected_poster, image = load_poster(image_path)
    lines = load_labels(label_path, selected_poster)
    image, aoi_boxes = draw_aoi_boxes(image, lines, classes)
    attention_scores = {}

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")

    fig = plt.gcf()
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}")