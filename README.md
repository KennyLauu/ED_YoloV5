# ED_YoloV5
Pytorch Yolo ImageEncryption  

This is a regional encryption method using yolov5. We can automatically or manually choose the region of interest to encrypt those images. In practical, we only need to encrypt some areas rather than the whole picture, so by using yolov5, we can save much time by encrypting the entire image and demonstrating that we can encrypt certain areas quickly and accurately by our algorithm. 

## Execute  

You could download it by:  
```powershell
git clone https://github.com/TZheLiu/ED_YoloV5.git  
cd ED_YoloV5
python ./DetectEncry.py
```  

If you need the latest code, you could switch the branch by `git swithch dev`  

> Please modified the location of image in code while you running this code.
> And Remember to download weight file in github, put it in weights/*

## Note  

We made some changes to the yolov5 code. To protect the user's information in pictures, we encrypt some areas of the image as large as possible, so we propose the ? method to choose the max bounding box rather than the max confidence. And use the mask to get the areas of intersection in the image.  

