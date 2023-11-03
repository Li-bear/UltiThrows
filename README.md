# Application Functionality - UltiThrows

**Overall Description:**

UltiThrows is a project aimed at analyzing ultimate frisbee throws through video input. The application leverages cutting-edge technology to extract statistics from throws, identify the specific throw type (backhand and forehand), and deliver valuable feedback to help players improve their performance.

The project is inspired by video analysis techniques used in sports like tennis, as seen in the following video: [Tennis Video Analysis](https://www.youtube.com/watch?v=_-E1tJ9sDF8).

**Specific Functions of the Target App:**

1. Implement pose estimation using MediaPipe for video uploads.
2. Extract and store pose data for analysis.
3. Apply frisbee tracking using YOLO technique.
4. Store frisbee's motion data for further analysis.
5. Define and calculate goal metrics, such as distance and accuracy.
6. Generate a summary of user performance based on goals and throws.
7. Collect labeled data for various types of ultimate frisbee throws.
8. Integrate Plotly to create interactive visualizations for statistics.
9. Enable frame-by-frame video playback for in-depth analysis.
10. Train machine learning models to classify different throw types.

# Description of the Technology

## Programming Environments

- **Jupyter Notebooks:** Used for developing and fine-tuning computer vision and analysis algorithms.
- **PyCharm Community Edition:** Utilized for writing and debugging application code.

## Programming Languages

- **Python:** The primary programming language for application development.
- **JavaScript:** Used for creating the frontend web application.

## Frameworks

- **MediaPipe:** The core framework for real-time analysis of ultimate frisbee throws.

## Libraries

- **NumPy:** Utilized for numerical computations and data manipulation.
- **OpenCV:** Employed for video processing and computer vision tasks.
- **Plotly:** Used for creating interactive charts and visualizations.

# Database

**Videos Metadata Table**: This database table stores metadata about each video, including columns like:

- Video ID: A unique identifier for each video.
- User ID: The ID of the user who uploaded the video.
- Video File Name: The name or path of the video file on the server.
- Upload Timestamp: The date and time when the video was uploaded.
- Analysis Status: A flag indicating whether the video has been analyzed.
- Analysis Results: Analysis results stored as structured data (e.g., JSON or related data formats).
- Download Count: A count of how many times the video has been downloaded.

# App Architecture

The client platform serves as the user-facing interface on the website, facilitating communication with the server responsible for video processing. It provides features such as video upload, real-time playback, and interactive visualizations to ensure seamless and engaging user interactions. Through this platform, users can track their throwing motion and frisbee trajectory, generate personalized statistics, and receive feedback on their performance. It plays a pivotal role in securely transmitting user input and video data to the server for in-depth analysis, ultimately enhancing the user experience by providing insights into ultimate frisbee performance and skill improvement.

# Technology Testing

- Unit testing of the computer vision pipeline in isolation.

# Methodology

The proposed methodology follows an Agile development approach, with the following sprint goals:

**Sprint 1**: Project Initialization and development environment setup. 
**Sprint 2:** Video input processing for user pose. 
**Sprint 3:** Video input processing for frisbee movement. 
**Sprint 4:** Statistics generation. 
**Sprint 5:** Throw classification. 
**Sprint 6:** Frontend Development.