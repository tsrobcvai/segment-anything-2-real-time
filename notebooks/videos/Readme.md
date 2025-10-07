# Get the first frame of the video
``` 
python save_first_frame.py            # uses merged_webcam_2.mp4 -> merged_webcam_2.png
# or
python save_first_frame.py merged_webcam_2.mp4 first_frame.jpg
``` 
# Generate mask for the first frame
``` 
python polygon_mask_tool.py path/to/your_image.jpg
#or specify output name (PNG will be appended):
python polygon_mask_tool.py input.jpg my_mask.png
```