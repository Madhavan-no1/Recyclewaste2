from bing_image_downloader import downloader

# Using double backslashes to handle the Windows file path correctly
downloader.download(
    "glass", 
    limit=100,  
    output_dir='C:\\Users\\rathn\\OneDrive\\Documents\\dataset-resized\\dataset-resized\\glass', 
    adult_filter_off=True, 
    force_replace=False, 
    timeout=60, 
    verbose=True
)
