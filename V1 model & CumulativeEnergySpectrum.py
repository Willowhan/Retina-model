import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_glop_energy(frame, frequencies, thetas, sigma):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
    gray = gray.astype(np.float32)
    energy = 0
    '''
    Convert each frame to grayscale
    Create filters in every direction and at every frequency
    np.fft.fft2(gray):The gray image is transformed from space domain to frequency domain by two-dimensional Fourier transform
    np.fft.fft2(gray) * g_filter: Fourier transform image * filter
    np.fft.ifft2(...): Convert the image from the frequency domain back to the spatial domain
    np.abs(filtered) ** 2: 
    '''
    for fk in frequencies:  # Traverse frequency and Angle
        for theta_i in thetas:
            g_filter = create_glop_filter(gray.shape, fk, theta_i, sigma)
            filtered = np.fft.ifft2(np.fft.fft2(gray) * g_filter)
            energy += np.sum(np.abs(filtered) ** 2) # The energy of all the pixels
    return energy


def create_glop_filter(shape, fk, theta_i, sigma):
    rows, cols = shape
    #print(rows,cols,sigma)
    fx = np.fft.fftfreq(cols)  # Calculate frequency axes fx and fy.
    fy = np.fft.fftfreq(rows)
    X, Y = np.meshgrid(fx, fy) # Generate frequency grids X and Y
    f = np.sqrt(X ** 2 + Y ** 2) # Calculate the radial frequency f
    theta = np.arctan2(Y, X)

    radial = (np.log(f / fk) / (2 * sigma ** 2)) ** 2
    angular = (1 + np.cos(theta - theta_i)) ** 50 / 2

    filter = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-radial) * angular
    return filter


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return []

    energies = []
    frequencies = [0.1, 0.2, 0.3] # More frequency values can be added to capture more scale information, e.g. [0.1, 0.2, 0.3] More frequency More precise
    thetas = [0, np.pi / 4, np.pi / 2] # [0, np.pi / 4, np.pi / 2][0, np.pi/6, np.pi/3, np.pi/2]
    sigma = 1.0
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        energy = compute_glop_energy(frame, frequencies, thetas, sigma) # calculates the energy of the current frame
        energies.append(energy)  # Add the result to the energies list
        if frame_index % 10 == 0:  # Print every 10 frames for debugging
            print(f"Processed frame {frame_index}, Energy: {energy}")  # Every 10 frames, the index and energy value of the currently processed frame is printed
    cap.release()
    return energies


# Provide the path to your video file
video_path = 'E:\\dissertation\\simulation\\blink.mp4'
energies = process_video(video_path)

if energies:
    plt.figure(figsize=(9.6, 5.4))
    plt.plot(energies, label='Global Spectrum Energy')
    plt.xlabel('Frame')
    plt.ylabel('Energy')
    plt.title('Global Spectrum Energy Over Frames')
    plt.legend()
    plt.show()
else:
    print("No energies computed.")
