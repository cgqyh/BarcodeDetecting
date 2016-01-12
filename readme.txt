用 Python 和 OpenCV 检测图片上的的条形码

http://www.cnblogs.com/pmars/p/4143158.html

这篇博文的目的是应用计算机视觉和图像处理技术，展示一个条形码检测的基本实现。我所实现的算法本质上基于StackOverflow 上的这个问题，浏览代码之后，我提供了一些对原始算法的更新和改进。

首先需要留意的是，这个算法并不是对所有条形码有效，但会给你基本的关于应用什么类型的技术的直觉。

假设我们要检测下图中的条形码：



图1：包含条形码的示例图片

现在让我们开始写点代码，新建一个文件，命名为detect_barcode.py，打开并编码：

1 # import the necessary packages
2 import numpy as np
3 import argparse
4 import cv2
5
6 # construct the argument parse and parse the arguments
7 ap = argparse.ArgumentParser()
8 ap.add_argument("-i", "--image", required = True, help = "path to the image file")
9 args = vars(ap.parse_args())
我们首先做的是导入所需的软件包，我们将使用NumPy做数值计算，argparse用来解析命令行参数，cv2是OpenCV的绑定。

然后我们设置命令行参数，我们这里需要一个简单的选择，–image是指包含条形码的待检测图像文件的路径。

现在开始真正的图像处理：

11 # load the image and convert it to grayscale
12 image = cv2.imread(args["image"])
13 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
14
15 # compute the Scharr gradient magnitude representation of the images
16 # in both the x and y direction
17 gradX = cv2.Sobel(gray, ddepth = cv2.cv.CV_32F, dx = 1, dy = 0, ksize = -1)
18 gradY = cv2.Sobel(gray, ddepth = cv2.cv.CV_32F, dx = 0, dy = 1, ksize = -1)
19
20 # subtract the y-gradient from the x-gradient
21 gradient = cv2.subtract(gradX, gradY)
22 gradient = cv2.convertScaleAbs(gradient)
12~13行：从磁盘载入图像并转换为灰度图。

17~18行：使用Scharr操作（指定使用ksize = -1）构造灰度图在水平和竖直方向上的梯度幅值表示。

21~22行：Scharr操作之后，我们从x-gradient中减去y-gradient，通过这一步减法操作，最终得到包含高水平梯度和低竖直梯度的图像区域。

上面的gradient表示的原始图像看起来是这样的：



图:2：条形码图像的梯度表示

注意条形码区域是怎样通过梯度操作检测出来的。下一步将通过去噪仅关注条形码区域。

24 # blur and threshold the image
25 blurred = cv2.blur(gradient, (9, 9))
26 (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
25行：我们要做的第一件事是使用9*9的内核对梯度图进行平均模糊，这将有助于平滑梯度表征的图形中的高频噪声。

26行：然后我们将模糊化后的图形进行二值化，梯度图中任何小于等于255的像素设为0（黑色），其余设为255（白色）。

模糊并二值化后的输出看起来是这个样子：



图3：二值化梯度图以此获得长方形条形码区域的粗略近似

然而，如你所见，在上面的二值化图像中，条形码的竖杠之间存在缝隙，为了消除这些缝隙，并使我们的算法更容易检测到条形码中的“斑点”状区域，我们需要进行一些基本的形态学操作：

28 # construct a closing kernel and apply it to the thresholded image
29 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
30 closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
29行：我们首先使用cv2.getStructuringElement构造一个长方形内核。这个内核的宽度大于长度，因此我们可以消除条形码中垂直条之间的缝隙。

30行：这里进行形态学操作，将上一步得到的内核应用到我们的二值图中，以此来消除竖杠间的缝隙。

现在，你可以看到这些缝隙相比上面的二值化图像基本已经消除：



图4：使用形态学中的闭运算消除条形码竖条之间的缝隙

当然，现在图像中还有一些小斑点，不属于真正条形码的一部分，但是可能影响我们的轮廓检测。

让我们来消除这些小斑点：

32 # perform a series of erosions and dilations
33 closed = cv2.erode(closed, None, iterations = 4)
34 closed = cv2.dilate(closed, None, iterations = 4)
我们这里所做的是首先进行4次腐蚀（erosion），然后进行4次膨胀（dilation）。腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，而膨胀操作将使剩余的白色像素扩张并重新增长回去。

如果小斑点在腐蚀操作中被移除，那么在膨胀操作中就不会再出现。

经过我们这一系列的腐蚀和膨胀操作，可以看到我们已经成功地移除小斑点并得到条形码区域。



图5：应用一系列的腐蚀和膨胀来移除不相关的小斑点

最后，让我们找到图像中条形码的轮廓：

36 # find the contours in the thresholded image, then sort the contours
37 # by their area, keeping only the largest one
38 (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
39  cv2.CHAIN_APPROX_SIMPLE)
40 c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
41
42 # compute the rotated bounding box of the largest contour
43 rect = cv2.minAreaRect(c)
44 box = np.int0(cv2.cv.BoxPoints(rect))
45
46 # draw a bounding box arounded the detected barcode and display the
47 # image
48 cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
49 cv2.imshow("Image", image)
50 cv2.waitKey(0)
38~40行：幸运的是这一部分比较容易，我们简单地找到图像中的最大轮廓，如果我们正确完成了图像处理步骤，这里应该对应于条形码区域。

43~44行：然后我们为最大轮廓确定最小边框

48~50行：最后显示检测到的条形码

正如你在下面的图片中所见，我们已经成功检测到了条形码：



图6：成功检测到示例图像中的条形码

下一部分，我们将尝试更多图像。

成功的条形码检测

要跟随这些结果，请使用文章下面的表单去下载本文的源码以及随带的图片。

一旦有了代码和图像，打开一个终端来执行下面的命令：

1
$ python detect_barcode.py --image images/barcode_02.jpg


图7：使用OpenCV检测图像中的一个条形码

检测椰油瓶子上的条形码没有问题。

让我们试下另外一张图片：

1
$ python detect_barcode.py --image images/barcode_03.jpg


图8：使用计算机视觉检测图像中的一个条形码

我们同样能够在上面的图片中找到条形码。

关于食品的条形码检测已经足够了，书本上的条形码怎么样呢：

1
$ python detect_barcode.py --image images/barcode_04.jpg


图9：使用Python和OpenCV检测书本上的条形码

没问题，再次通过。

那包裹上的跟踪码呢？

1
$ python detect_barcode.py --image images/barcode_05.jpg


图10：使用计算机视觉和图像处理检测包裹上的条形码

我们的算法再次成功检测到条形码。

最后，我们再尝试一张图片，这个是我最爱的意大利面酱—饶氏自制伏特加酱（Rao’s Homemade Vodka Sauce）:

1
$ python detect_barcode.py --image images/barcode_06.jpg


图11：使用Python和Opencv很容易检测条形码

我们的算法又一次检测到条形码！

总结

这篇博文中，我们回顾了使用计算机视觉技术检测图像中条形码的必要步骤，使用Python编程语言和OpenCV库实现了我们的算法。

算法概要如下：

计算x方向和y方向上的Scharr梯度幅值表示
将x-gradient减去y-gradient来显示条形码区域
模糊并二值化图像
对二值化图像应用闭运算内核
进行系列的腐蚀、膨胀
找到图像中的最大轮廓，大概便是条形码
需要注意的是，该方法做了关于图像梯度表示的假设，因此只对水平条形码有效。

如果你想实现一个更加鲁棒的条形码检测算法，你需要考虑图像的方向，或者更好的，应用机器学习技术如Haar级联或者HOG + Linear SVM去扫描图像条形码区域。