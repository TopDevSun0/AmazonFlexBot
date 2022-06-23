// AmazonFlexBot.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <Windows.h>
#include <vector>


// global variables
const int MIN_CONTOUR_AREA = 5;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;
const int MIN_PRICE = 120;

void getMousePos() {
    POINT c;
    GetCursorPos(&c);
    std::cout << c.x << " " << c.y << std::endl;
    Sleep(200);
}


void train_model() {
    cv::Mat imgTrainingNumbers;         // input image
    cv::Mat imgGrayscale;               // 
    cv::Mat imgBlurred;                 // declare various images
    cv::Mat imgThresh;                  //
    cv::Mat imgThreshCopy;              //

    std::vector<std::vector<cv::Point> > ptContours;        // declare contours vector
    std::vector<cv::Vec4i> v4iHierarchy;                    // declare contours hierarchy

    cv::Mat matClassificationInts;      // these are our training classifications, note we will have to perform some conversions before writing to file later

                                        // these are our training images, due to the data types that the KNN object KNearest requires, we have to declare a single Mat,
                                        // then append to it as though it's a vector, also we will have to perform some conversions before writing to file later
    cv::Mat matTrainingImagesAsFlattenedFloats;

    // possible chars we are interested in are digits 0 through 9 and capital letters A through Z, put these in vector intValidChars
    std::vector<int> intValidChars = { (char)46 ,'0', '1', '2', '3', '4', '5', '6', '7', '8', '9','$' };
        //'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        //'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        //'U', 'V', 'W', 'X', 'Y', 'Z' , '$', '.'};

    imgTrainingNumbers = cv::imread("train_img.png");          // read in training numbers image

    if (imgTrainingNumbers.empty()) {                               // if unable to open image
        std::cout << "error: image not read from file\n\n";         // show error message on command line
        return;                                                  // and exit program
    }

    cv::cvtColor(imgTrainingNumbers, imgGrayscale, cv::COLOR_BGR2GRAY);        // CV_BGR2GRAY convert to grayscale

    cv::GaussianBlur(imgGrayscale,              // input image
        imgBlurred,                             // output image
        cv::Size(3, 3),                         // smoothing window width and height in pixels
        2);                                     // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

                                                // filter image from grayscale to black and white
    cv::adaptiveThreshold(imgBlurred,           // input image
        imgThresh,                              // output image
        255,                                    // make pixels that pass the threshold full white
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,         // use gaussian rather than mean, seems to give better results
        cv::THRESH_BINARY_INV,                  // invert so foreground will be white, background will be black
        3,                                     // size of a pixel neighborhood used to calculate threshold value
        2);                                     // constant subtracted from the mean or weighted mean

    cv::imshow("imgThresh", imgThresh);         // show threshold image for reference

    imgThreshCopy = imgThresh.clone();          // make a copy of the thresh image, this in necessary b/c findContours modifies the image

    cv::findContours(imgThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
        ptContours,                             // output contours
        v4iHierarchy,                           // output hierarchy
        cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
        cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points

    for (int i = 0; i < ptContours.size(); i++) {                           // for each contour
        if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {                // if contour is big enough to consider
            cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                // get the bounding rect

            cv::rectangle(imgTrainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);      // draw red rectangle around each contour as we ask user for input

            cv::Mat matROI = imgThresh(boundingRect);           // get ROI image of bounding rect

            cv::Mat matROIResized;
            cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

            cv::imshow("matROI", matROI);                               // show ROI image for reference
            cv::imshow("matROIResized", matROIResized);                 // show resized ROI image for reference
            cv::imshow("imgTrainingNumbers", imgTrainingNumbers);       // show training numbers image, this will now have red rectangles drawn on it

            int intChar = cv::waitKey(0);           // get key press

            if (intChar == 27) {        // if esc key was pressed
                return;              // exit program
            }
            else if (std::find(intValidChars.begin(), intValidChars.end(), intChar) != intValidChars.end()) {     // else if the char is in the list of chars we are looking for . . .

                matClassificationInts.push_back(intChar);       // append classification char to integer list of chars

                cv::Mat matImageFloat;                          // now add the training image (some conversion is necessary first) . . .
                matROIResized.convertTo(matImageFloat, CV_32FC1);       // convert Mat to float

                cv::Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);       // flatten

                matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);       // add to Mat as though it was a vector, this is necessary due to the
                                                                                            // data types that KNearest.train accepts
            }   // end if
        }   // end if
    }   // end for

    std::cout << "training complete\n\n";

    // save classifications to file ///////////////////////////////////////////////////////

    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);           // open the classifications file

    if (fsClassifications.isOpened() == false) {                                                        // if the file was not opened successfully
        std::cout << "error, unable to open training classifications file, exiting program\n\n";        // show error message
        return;                                                                                      // and exit program
    }

    fsClassifications << "classifications" << matClassificationInts;        // write classifications into classifications section of classifications file
    fsClassifications.release();                                            // close the classifications file

                                                                            // save training images to file ///////////////////////////////////////////////////////

    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);         // open the training images file

    if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
        std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
        return;                                                                              // and exit program
    }

    fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;         // write training images into images section of images file
    fsTrainingImages.release();                                                 // close the training images file
}


class ContourWithData {
public:
    // member variables ///////////////////////////////////////////////////////////////////////////
    std::vector<cv::Point> ptContour;           // contour
    cv::Rect boundingRect;                      // bounding rect for contour
    float fltArea;                              // area of contour

                                                ///////////////////////////////////////////////////////////////////////////////////////////////
    bool checkIfContourIsValid() {                              // obviously in a production grade program
        if (fltArea < MIN_CONTOUR_AREA) return false;           // we would have a much more robust function for 
        return true;                                            // identifying if a contour is valid !!
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
        return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   // the contours from left to right
    }

};

float recogprice(cv::Mat matTestingNumbers, cv::Ptr<cv::ml::KNearest>  kNearest) {
    std::vector<ContourWithData> allContoursWithData;           // declare empty vectors,
    std::vector<ContourWithData> validContoursWithData;         // we will fill these shortly

    //cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            // instantiate the KNN object

    //                                                                            // finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
    //                                                                            // even though in reality they are multiple images / numbers
    //kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    // cv::Mat matTestingNumbers = cv::imread("75.png");            // read in the test numbers image
    //if (matTestingNumbers.empty()) {                                // if unable to open image
    //    std::cout << "error: image not read from file\n\n";         // show error message on command line
    //    return -1;                                                  // and exit program
    //}

    cv::Mat matGrayscale;           //
    cv::Mat matBlurred;             // declare more image variables
    cv::Mat matThresh;              //
    cv::Mat matThreshCopy;          //

    cv::cvtColor(matTestingNumbers, matGrayscale, cv::COLOR_BGR2GRAY);         // convert to grayscale

                                                                        // blur
    cv::GaussianBlur(matGrayscale,              // input image
        matBlurred,                // output image
        cv::Size(3, 3),            // smoothing window width and height in pixels
        2);                        // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

                                   // filter image from grayscale to black and white
    cv::adaptiveThreshold(matBlurred,                           // input image
        matThresh,                            // output image
        255,                                  // make pixels that pass the threshold full white
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,       // use gaussian rather than mean, seems to give better results
        cv::THRESH_BINARY_INV,                // invert so foreground will be white, background will be black
        3,                                   // size of a pixel neighborhood used to calculate threshold value
        2);                                   // constant subtracted from the mean or weighted mean

    matThreshCopy = matThresh.clone();              // make a copy of the thresh image, this in necessary b/c findContours modifies the image

    std::vector<std::vector<cv::Point> > ptContours;        // declare a vector for the contours
    std::vector<cv::Vec4i> v4iHierarchy;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)

    cv::findContours(matThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
        ptContours,                             // output contours
        v4iHierarchy,                           // output hierarchy
        cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
        cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points

    for (int i = 0; i < ptContours.size(); i++) {               // for each contour
        ContourWithData contourWithData;                                                    // instantiate a contour with data object
        contourWithData.ptContour = ptContours[i];                                          // assign contour to contour with data
        contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);         // get the bounding rect
        contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);               // calculate the contour area
        allContoursWithData.push_back(contourWithData);                                     // add contour with data object to list of all contours with data
    }

    for (int i = 0; i < allContoursWithData.size(); i++) {                      // for all contours
        if (allContoursWithData[i].checkIfContourIsValid()) {                   // check if valid
            validContoursWithData.push_back(allContoursWithData[i]);            // if so, append to valid contour list
        }
    }
    // sort contours from left to right
    std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

    std::string strFinalString;         // declare final string, this will have the final number sequence by the end of the program

    //for (int i = 0; i < validContoursWithData.size(); i++) {            // for each contour
    for(const auto& i: validContoursWithData){
                                                                        // draw a green rect around the current char
        //cv::rectangle(matTestingNumbers,                            // draw rectangle on original image
        //    validContoursWithData[i].boundingRect,        // rect to draw
        //    cv::Scalar(0, 255, 0),                        // green
        //    1);                                           // thickness

        cv::Mat matROI = matThresh(i.boundingRect);          // get ROI image of bounding rect

        cv::Mat matROIResized;
        cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

        cv::Mat matROIFloat;
        matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest

        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

        cv::Mat matCurrentChar(0, 0, CV_32F);

        kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     // finally we can call find_nearest !!!

        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

        strFinalString = strFinalString + char(int(fltCurrentChar));        // append current char to full string
    }
    float price = 0;
    try {
        if (strFinalString.length() < 4)
            price = 0;
        else if (strFinalString[1] == '$')
            price = stof(strFinalString.substr(2, strFinalString.length()));
        else if (strFinalString[2] == '$')
            price = stof(strFinalString.substr(3, strFinalString.length()));
        else
            price = 0;
    }
    catch(...){
        price = 0;
    }
    
    //if (price > 150) {
    //    cout << "\n\ntarget price " << price << endl;
    //}
    std::cout << "Price read = " << "$" << price << "\n\n";       // show the full string


    //cv::imshow("matTestingNumbers", matTestingNumbers);     // show input image with green boxes drawn around found digits
    
    // cv::waitKey(0);                                         // wait for user key press
    return price;
}


cv::Mat getwindow(HWND hWND, int y) {
    // HRGN hRGN = CreateRectRgn(683, 284, 768, 1313);
    // HDC windowHandler = GetDCEx(hWND, hRGN, DCX_WINDOW);
    HDC windowHandler = GetDC(hWND);
    HDC memoryDevice = CreateCompatibleDC(windowHandler);
    RECT windowRect;
    GetClientRect(hWND, &windowRect);
    int height = windowRect.bottom; // 1392
    int width = windowRect.right;   // 798


    HBITMAP bitmap = CreateCompatibleBitmap(windowHandler, width, height);
    SelectObject(memoryDevice, bitmap);
    // bitblock of price range only
    // BitBlt(memoryDevice, 0, 0, 100, 50, windowHandler, 660, 300+y, SRCCOPY);
    BitBlt(memoryDevice, 0, 0, 100, 90, windowHandler, width*0.8271, height*0.21552 + y, SRCCOPY);
    BITMAPINFOHEADER bi;
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height;  // must be -height, otherwise is upside down
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 1;
    bi.biYPelsPerMeter = 2;
    bi.biClrUsed = 3;
    bi.biClrImportant = 4;

    cv::Mat mat = cv::Mat(height, width, CV_8UC4);
    GetDIBits(windowHandler, bitmap, 0, height, mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    DeleteObject(bitmap);
    DeleteDC(memoryDevice);
    // DeleteObject(hRGN);
    ReleaseDC(hWND, windowHandler);
   
    return mat;
}

void readScreen(cv::Ptr<cv::ml::KNearest>  kNearest) {
    LPCWSTR windowTitle = L"BlueStacks App Player";
    HWND hWND = FindWindow(NULL, windowTitle);
    RECT windowRect;
    GetWindowRect(hWND, &windowRect);
    int height = windowRect.bottom; 
    int width = windowRect.right;   
    // cv::namedWindow("AmazonFlex", cv::WINDOW_NORMAL);
    while (!hWND) {
        system("cls");
        std::cout << "Not found amazon flex app window!" << std::endl;
        hWND = FindWindow(NULL, windowTitle);
        Sleep(100);
    }
    const int refresh_x = width / 2;
    const int refresh_y = height * 0.97;
    const int swipe_x = width * 0.1;
    const int swipe_y = height * 0.97;
    const int end_swipe_x = swipe_x + width * 0.6;
    const int end_of_screen = height * 0.5; // adjust to look num of offers.
    const int price_pos_x = width * 0.8522;
    const int price_pos_y = height * 0.215;
    int yy = 0;
    while (true) {
        /*HWND temp = GetForegroundWindow();
        if (temp != hWND) {
            Sleep(10);
            continue;
        }*/
        if (GetKeyState(VK_RCONTROL)) {
            return;
        }
        SetCursorPos(refresh_x, refresh_y);
        mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
        mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);

        cv::Mat target = getwindow(hWND,yy);
        cv::Mat background;
        target.copyTo(background);
        cv::cvtColor(target, target, cv::COLOR_BGR2HSV);
        // cv::imshow("AmazonFlex", background);
        float price = recogprice(background, kNearest);
        if (price >= MIN_PRICE) { 
            //POINT offer;
            //offer.x = price_pos_x;
            //offer.y = yy + price_pos_y;
            SetCursorPos(price_pos_x, yy + price_pos_y);
            mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
            SetCursorPos(swipe_x, swipe_y);
            Sleep(90);
            mouse_event(MOUSEEVENTF_LEFTDOWN, swipe_x, swipe_y, 0, 0);
            Sleep(80);
            SetCursorPos(end_swipe_x, swipe_y);
            Sleep(70);
            mouse_event(MOUSEEVENTF_LEFTUP, end_swipe_x, swipe_y, 0, 0);
        }

        yy += 120;
        if (yy > end_of_screen) {
            yy = 0;
        }
    }
}

int main()
{
    cv::Mat matClassificationInts;      // we will read the classification numbers into this variable as though it is a vector
    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        // open the classifications file

    if (fsClassifications.isOpened() == false) {                                                    // if the file was not opened successfully
        std::cout << "error, unable to open training classifications file, exiting program\n\n";    // show error message
        return -1;                                                                                  // and exit program
    }

    fsClassifications["classifications"] >> matClassificationInts;      // read classifications section into Mat classifications variable
    fsClassifications.release();                                        // close the classifications file

                                                                        // read in training images ////////////////////////////////////////////////////////////

    cv::Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);          // open the training images file

    if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
        std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
        return -1;                                                                              // and exit program
    }

    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
    fsTrainingImages.release();                                                 // close the traning images file
    cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            // instantiate the KNN object

                                                                                // finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
                                                                                // even though in reality they are multiple images / numbers
    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    //train_model();
    readScreen(kNearest);

    return 0;
}


