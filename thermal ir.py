import cv2
import numpy as np


def capture_thermal_image():
    # Capture the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return None
    # Convert the frame to grayscale for simulating thermal imaging
    thermal_data = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return thermal_data


def display_thermal_image(thermal_data):
    # Normalize the image to the range of 0 to 255 for visualization
    thermal_normalized = cv2.normalize(thermal_data, None, 0, 255, cv2.NORM_MINMAX)
    thermal_display = np.uint8(thermal_normalized)

    # Apply a colormap to simulate thermal imaging
    thermal_colormap = cv2.applyColorMap(thermal_display, cv2.COLORMAP_JET)

    # Detect heat signatures by thresholding
    heat_signatures = detect_heat_signatures(thermal_data)

    # Highlight detected heat signatures in the colormap and show temperature
    for contour in heat_signatures:
        # Draw contour in red
        cv2.drawContours(thermal_colormap, [contour], -1, (0, 0, 255), 2)

        # Calculate the average temperature in the heat signature region
        temperature = calculate_average_temperature(thermal_data, contour)

        # Convert the temperature to a string and display it
        x, y, w, h = cv2.boundingRect(contour)
        text = f"{temperature:.1f}°C"
        cv2.putText(thermal_colormap, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the processed image in a window
    cv2.imshow("Thermal Image", thermal_colormap)


def detect_heat_signatures(thermal_data, threshold=180):
    # Threshold to simulate heat signature detection (values above threshold are considered "hot")
    _, thresh = cv2.threshold(thermal_data, threshold, 255, cv2.THRESH_BINARY)

    # Find contours of the "hot" areas (areas with high temperature)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise) and return the detected heat signature contours
    heat_signatures = [contour for contour in contours if cv2.contourArea(contour) > 500]

    return heat_signatures


def calculate_average_temperature(thermal_data, contour):
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Extract the region of interest (ROI) from the thermal data
    roi = thermal_data[y:y + h, x:x + w]

    # Calculate the average pixel value (which we'll treat as the average temperature)
    average_pixel_value = np.mean(roi)

    # Map the pixel value to a temperature (this is a linear approximation)
    # Assuming 0 intensity -> 20°C and 255 intensity -> 40°C
    min_temp = 20  # Minimum temperature (in °C)
    max_temp = 40  # Maximum temperature (in °C)
    temperature = min_temp + (average_pixel_value / 255) * (max_temp - min_temp)

    return temperature


def main():
    global cap
    # Open the webcam (device 0 is typically the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    while True:
        thermal_data = capture_thermal_image()
        if thermal_data is not None:
            display_thermal_image(thermal_data)

        # Check for the 'q' key to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
