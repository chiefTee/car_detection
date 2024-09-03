from pytube import YouTube
from urllib.error import HTTPError

try:
    # Example YouTube video URL
    video_url = 'https://youtu.be/MNn9qKG2UFI?si=Pt6RE8dt17OV67ne'
    
    # Create a YouTube object
    yt = YouTube(video_url)

    # Choose the highest resolution stream available
    stream = yt.streams.get_highest_resolution()

    # Specify the download directory (e.g., Downloads folder)
    download_directory = r'C:\Users\Industry Expert\Desktop\Africa agility model deployment'

    # Download the video
    stream.download(output_path=download_directory)

    print(f"Video downloaded successfully to {download_directory}")

except HTTPError as e:
    print(f"HTTP Error: {e.code} - {e.reason}")
except Exception as e:
    print(f"An error occurred: {e}")
