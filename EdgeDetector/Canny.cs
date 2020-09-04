using System;
using System.Drawing.Imaging;
using System.Drawing;

namespace EdgeDetector
{
    public struct InspectionArea
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
    }

    /// <summary>
    /// Canny edge detection class
    /// </summary>
    public sealed class Canny : IDisposable
    {
        // Image data
        private readonly int _imgWidth, _imgHeight;
        private readonly InspectionArea _inspectArea;
        private Bitmap bmp;

        //Gaussian Kernel Data
        private int kernelSize = 5;
        private float sigma = 1.4f;

        // Canny Edge Detection Data
        private int[,] edgePoints;
        private bool[,] visitedMap;

        /// <summary>
        /// Get and Set the maximum hysteresis threshold
        /// </summary>
        public int MaxHysteresisThresh { get; set; }
        /// <summary>
        /// Get and Set the minimum hysterisis thereshold
        /// </summary>
        public int MinHysteresisThresh { get; set; }

        public int[,] EdgeMap { get; set; }
        private int[,] filteredImage;

        // Cleanup
        private bool disposed = false;

        /// <summary>
        /// Initializes a new instance of the Canny class from the specified file.
        /// - Use this constructor when you need to provide initial hysteresis threshold.
        /// </summary>
        /// <param name="input">Bitmap</param>
        /// <param name="Th">Maximum hysteresis threshold</param>
        /// <param name="Tl">Minimum hysteresis threshold</param>
        /// <param name="GaussianMaskSize">Gaussian filter mask size (odd number only)</param>
        /// <param name="SigmaforGaussianKernel">Standard deviation for the Gaussian filter</param>
        public Canny(Bitmap input, int Th, int Tl, int GaussianMaskSize, float SigmaforGaussianKernel)
        {            
            // Image
            bmp = input;
            _imgWidth = bmp.Width;
            _imgHeight = bmp.Height;
            // Area
            _inspectArea.Width = _imgWidth;
            _inspectArea.Height = _imgHeight;

            // Gaussian and Canny Parameters            
            MaxHysteresisThresh = Th;
            MinHysteresisThresh = Tl;
            kernelSize = GaussianMaskSize;
            sigma = SigmaforGaussianKernel;

            InitializeImageBuffer(input);
            GetGrayImage();
            DetectCannyEdges();
            return;
        }

        /// <summary>
        /// Initializes a new instance of the Canny class from the specified file.
        /// - Use this constructor when you need to specify custom inspection area.
        /// </summary>
        /// <param name="input">Bitmap input</param>
        /// <param name="area">Inspection area</param>
        /// <param name="GaussianMaskSize">Gaussian filter mask size (odd number only)</param>
        /// <param name="SigmaforGaussianKernel">Standard deviation for the Gaussian filter</param>
        public Canny(Bitmap input, InspectionArea area, int GaussianMaskSize, float SigmaforGaussianKernel)
        {
            // Area
            _inspectArea = area;
            // Image
            bmp = input;
            _imgWidth = bmp.Width;
            _imgHeight = bmp.Height;

            // Gaussian and Canny Parameters
            kernelSize = GaussianMaskSize;
            sigma = SigmaforGaussianKernel;

            InitializeImageBuffer(input);
            GetGrayImage();
            DetectCannyEdges();
            return;
        }

        private void InitializeImageBuffer(Bitmap input)
        {                        
            EdgeMap = new int[_imgWidth, _imgHeight];
        }

        ~Canny()
        {
            //GC triggered the cleanup
            CleanUp(false);
        }

        private void CleanUp(bool disposing)
        {
            if (!this.disposed)
                if (disposing)
                    bmp.Dispose();

            disposed = true;
        }

        /// <summary>
        /// Dispose Canny object from memory
        /// </summary>
        public void Dispose()
        {
            CleanUp(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Get final canny image
        /// </summary>
        /// <returns>
        /// Bitmap image
        /// </returns>
        public Bitmap GetCannyImage()
        {
            return BufferToImage(EdgeMap);
        }

        /// <summary>
        /// Get filtered image
        /// </summary>
        /// <returns>
        /// Bitmap image
        /// </returns>
        public Bitmap GetFilteredImage()
        {
            return BufferToImage(filteredImage);
        }

        private Bitmap BufferToImage(int[,] buffer)
        {
            int W = buffer.GetLength(0);
            int H = buffer.GetLength(1);
            Bitmap bmp = new Bitmap(W, H);
            Rectangle rect = new Rectangle(0, 0, W, H);
            BitmapData bitmapData = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

            unsafe
            {
                byte* imgPtr = (byte*)bitmapData.Scan0;

                for (int y = 0; y < bitmapData.Height; y++)
                {
                    for (int x = 0; x < bitmapData.Width; x++)
                    {
                        imgPtr[0] = (byte)buffer[x, y];
                        imgPtr[1] = (byte)buffer[x, y];
                        imgPtr[2] = (byte)buffer[x, y];
                        imgPtr[3] = 0xFF;
                        // point to the next pixel (4 bytes per pixel)
                        imgPtr += 4;
                    }
                }
            }

            bmp.UnlockBits(bitmapData);
            return bmp;
        }

        private void DetectCannyEdges()
        {
            int kernelRadius = kernelSize / 2;

            // 1. Noise reduction
            // 1a. Perform gaussian filtering
            filteredImage = ApplyGaussianFilterTo(GetGrayImage());
            // 1b. Perform adaptive filtering
            double[,] dx1 = ComputeGradient(filteredImage, kernelRadius, HorizontalGradient);
            double[,] dy1 = ComputeGradient(filteredImage, kernelRadius, VerticalGradient);
            double[,] gradientMagnitude = GetGradientMagnitude(dx1, dy1);
            double[,] weight = GetWeight(gradientMagnitude, 1f, kernelRadius);
            filteredImage = ApplyAdaptiveFilterTo(filteredImage, weight, kernelRadius);

            // 2. Gradient calculation
            double[,] dx = GetHorizontalDerivative(filteredImage);
            double[,] dy = GetVerticalDerivative(filteredImage);

            // 3. Non-maximum suppression
            int[,] nonMax = GetNonMaxSuppression(dx, dy, kernelRadius);

            // 4. Double thresholding
            SetHysteresisThreshold(nonMax);

            // 5. Egde tracking by hysteresis (remove false edge and connect the true edge)
            ApplyDoubleThresholding(nonMax, kernelRadius);
            TrackWeakEdge(kernelRadius);
        }

        private int[,] GetGrayImage()
        {
            int[,] grayImage = new int[bmp.Width, bmp.Height];
            Bitmap image = bmp;
            Rectangle rect = new Rectangle(0, 0, image.Width, image.Height);
            BitmapData bitmapData = image.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

            unsafe
            {
                byte* imgPtr = (byte*)bitmapData.Scan0;

                for (int y = 0; y < bitmapData.Height; y++)
                {
                    for (int x = 0; x < bitmapData.Width; x++)
                    {
                        grayImage[x, y] = (int)((imgPtr[0] + imgPtr[1] + imgPtr[2]) / 3.0);
                        // point to the next pixel (4 bytes per pixel)
                        imgPtr += 4;
                    }
                    //4 bytes per pixel
                    imgPtr += bitmapData.Stride - (bitmapData.Width * 4);
                }
            }

            image.UnlockBits(bitmapData);

            return grayImage;
        }

        private double[,] GetGaussianKernel()
        {
            double[,] kernel = new double[kernelSize, kernelSize];
            double sumTotal = 0;

            int kernelRadius = kernelSize / 2;
            double distance = 0;

            double calculatedEuler = 1.0 / (2.0 * Math.PI * sigma * sigma);

            // Calculate each kernel element
            for (int y = -kernelRadius; y <= kernelRadius; y++)
            {
                for (int x = -kernelRadius; x <= kernelRadius; x++)
                {
                    distance = ((x * x) + (y * y)) / (2.0 * sigma * sigma);

                    kernel[y + kernelRadius, x + kernelRadius] = calculatedEuler * Math.Exp(-distance);

                    sumTotal += kernel[y + kernelRadius, x + kernelRadius];
                }
            }

            // Correct the kernel values, ensuring the sum total of all kernel elements equate to 1
            for (int y = 0; y < kernelSize; y++)
            {
                for (int x = 0; x < kernelSize; x++)
                {
                    kernel[y, x] = kernel[y, x] * (1.0 / sumTotal);
                }
            }

            return kernel;
        }

        private int[,] ApplyGaussianFilterTo(int[,] grayscaleImage)
        {
            int[,] result = new int[_imgWidth, _imgHeight];
            int kernelRadius = kernelSize / 2;

            double sum = 0;

            // Get calculated gaussian mask
            double[,] GaussianMask = GetGaussianKernel();

            // Convolve input image with gaussian mask
            for (int x = kernelRadius; x < (_imgWidth - kernelRadius); x++)
            {
                for (int y = kernelRadius; y < (_imgHeight - kernelRadius); y++)
                {
                    sum = 0;
                    for (int kernelX = -kernelRadius; kernelX <= kernelRadius; kernelX++)
                    {
                        for (int kernelY = -kernelRadius; kernelY <= kernelRadius; kernelY++)
                        {
                            sum = sum + grayscaleImage[x + kernelX, y + kernelY] * GaussianMask[kernelRadius + kernelX, kernelRadius + kernelY];
                        }
                    }
                    result[x, y] = (int)Math.Round(sum);
                }
            }

            return result;
        }

        private double[,] ComputeGradient(int[,] data, int kernelRadius, Func<int[,], int, int, double> func)
        {
            double[,] result = new double[_imgWidth, _imgHeight];

            for (int x = kernelRadius; x < (_imgWidth - kernelRadius); x++)
            {
                for (int y = kernelRadius; y < (_imgHeight - kernelRadius); y++)
                {
                    result[x, y] = func(data, x, y);
                }
            }

            return result;
        }

        private double HorizontalGradient(int[,] imageBuffer, int x, int y)
        {
            return 0.5 * (imageBuffer[x + 1, y] - imageBuffer[x - 1, y]);
        }

        private double VerticalGradient(int[,] imageBuffer, int x, int y)
        {
            return 0.5 * (imageBuffer[x, y + 1] - imageBuffer[x, y - 1]);
        }

        private double[,] GetWeight(double[,] dX, float h, int kernelRadius)
        {
            double[,] result = new double[_imgWidth, _imgHeight];

            for (int x = kernelRadius; x < (_imgWidth - kernelRadius); x++)
            {
                for (int y = kernelRadius; y < (_imgHeight - kernelRadius); y++)
                {
                    result[x, y] = Math.Exp(-(Math.Sqrt(dX[x, y]) / (2 * h * h)));
                }
            }

            return result;
        }

        private int[,] ApplyAdaptiveFilterTo(int[,] grayscaleImage, double[,] weight, int kernelRadius)
        {
            int[,] result = new int[_imgWidth, _imgHeight];
            double sum, n;

            for (int x = kernelRadius; x < (_imgWidth - kernelRadius); x++)
            {
                for (int y = kernelRadius; y < (_imgHeight - kernelRadius); y++)
                {
                    sum = 0;
                    n = 0;
                    for (int i = -1; i <= 1; i++)
                    {
                        for (int j = -1; j <= 1; j++)
                        {
                            n += weight[x + i, y + j];
                            sum = sum + grayscaleImage[x + i, y + j] * weight[x + i, y + j];
                        }
                    }
                    result[x, y] = (int)(sum / n);
                }
            }

            return result;
        }

        private double[,] ConvolveFilter(int[,] imageBuf, int[,] filter)
        {
            int filterWidth = filter.GetLength(0);
            int filterHeight = filter.GetLength(1);

            int filterXRadius = filterWidth / 2;
            int filterYRadius = filterHeight / 2;

            double sum = 0;
            double[,] result = new double[_imgWidth, _imgHeight];

            for (int x = filterXRadius; x <= (_imgWidth - filterXRadius) - 1; x++)
            {
                for (int y = filterYRadius; y <= (_imgHeight - filterYRadius) - 1; y++)
                {
                    sum = 0;
                    for (int filterX = -filterXRadius; filterX <= filterXRadius; filterX++)
                    {
                        for (int filterY = -filterYRadius; filterY <= filterYRadius; filterY++)
                        {
                            sum = sum + imageBuf[x + filterX, y + filterY] * filter[filterXRadius + filterX, filterYRadius + filterY];
                        }
                    }
                    result[x, y] = sum;
                }
            }
            return result;
        }

        private double[,] GetHorizontalDerivative(int[,] filteredImage)
        {
            //Horizontal sobel mask
            int[,] Dx = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };

            //Convolve input image with sobel mask
            return ConvolveFilter(filteredImage, Dx);
        }

        private double[,] GetVerticalDerivative(int[,] filteredImage)
        {
            //Vertical sobel mask
            int[,] Dy = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

            //Convolve input image with sobel mask
            return ConvolveFilter(filteredImage, Dy);
        }

        private double[,] GetGradientMagnitude(double[,] DerivativeX, double[,] DerivativeY)
        {
            double[,] result = new double[_imgWidth, _imgHeight];

            //Compute the gradient magnitude based on derivatives in x and y:
            for (int x = 0; x < _inspectArea.Width; x++)
            {
                for (int y = 0; y < _inspectArea.Height; y++)
                {
                    result[x, y] = (float)Math.Sqrt((DerivativeX[x, y] * DerivativeX[x, y]) + (DerivativeY[x, y] * DerivativeY[x, y]));
                }
            }

            return result;
        }

        private int[,] GetNonMaxSuppression(double[,] DerivativeX, double[,] DerivativeY, int kernelRadius)
        {
            double[,] gradientMagnitude = GetGradientMagnitude(DerivativeX, DerivativeY);
            double[,] nonMax = new double[_imgWidth, _imgHeight];
            int[,] intBuffer = new int[_imgWidth, _imgHeight];

            // Prepare buffer for non maximum suppression result
            for (int x = 0; x < _inspectArea.Width; x++)
            {
                for (int y = 0; y < _inspectArea.Height; y++)
                {
                    nonMax[x, y] = gradientMagnitude[x, y];
                }
            }

            // Perform Non maximum suppression:
            double Tangent;

            for (int x = kernelRadius; x < (_inspectArea.Width - kernelRadius); x++)
            {
                for (int y = kernelRadius; y < (_inspectArea.Height - kernelRadius); y++)
                {
                    if (DerivativeX[x, y] == 0)
                        Tangent = 90d;
                    else
                        //rad to degree
                        Tangent = (float)(Math.Atan(DerivativeY[x, y] / DerivativeX[x, y]) * 180 / Math.PI);

                    //Horizontal Edge
                    if (((-22.5 < Tangent) && (Tangent <= 22.5)) || ((157.5 < Tangent) && (Tangent <= -157.5)))
                    {
                        if ((gradientMagnitude[x, y] < gradientMagnitude[x, y + 1]) || (gradientMagnitude[x, y] < gradientMagnitude[x, y - 1]))
                            nonMax[x, y] = 0;
                    }

                    //Vertical Edge
                    if (((-112.5 < Tangent) && (Tangent <= -67.5)) || ((67.5 < Tangent) && (Tangent <= 112.5)))
                    {
                        if ((gradientMagnitude[x, y] < gradientMagnitude[x + 1, y]) || (gradientMagnitude[x, y] < gradientMagnitude[x - 1, y]))
                            nonMax[x, y] = 0;
                    }

                    //+45 Degree Edge
                    if (((-67.5 < Tangent) && (Tangent <= -22.5)) || ((112.5 < Tangent) && (Tangent <= 157.5)))
                    {
                        if ((gradientMagnitude[x, y] < gradientMagnitude[x + 1, y - 1]) || (gradientMagnitude[x, y] < gradientMagnitude[x - 1, y + 1]))
                            nonMax[x, y] = 0;
                    }

                    //-45 Degree Edge
                    if (((-157.5 < Tangent) && (Tangent <= -112.5)) || ((22.5 < Tangent) && (Tangent <= 67.5)))
                    {
                        if ((gradientMagnitude[x, y] < gradientMagnitude[x + 1, y + 1]) || (gradientMagnitude[x, y] < gradientMagnitude[x - 1, y - 1]))
                            nonMax[x, y] = 0;
                    }

                    //Prepare non-max suppression buffer for post hysteresis
                    intBuffer[x, y] = (int)nonMax[x, y];
                }
            }

            return intBuffer;
        }

        private void SetHysteresisThreshold(int[,] postHysteresis)
        {
            byte[] hysteresisBuf = new byte[postHysteresis.Length];
            Buffer.BlockCopy(postHysteresis, 0, hysteresisBuf, 0, postHysteresis.Length);

            // Find Max in Post Hysteresis
            int max = 0;

            for (int i = 0; i < hysteresisBuf.Length; i++)
            {
                if (hysteresisBuf[i] > max) max = hysteresisBuf[i];
            }

            int[] histData = new int[max + 1];
            int ptr = 0;
            int index;

            // Clear buffer
            while (ptr < histData.Length) histData[ptr++] = 0;

            // Reset counter
            ptr = 0;

            while (ptr < hysteresisBuf.Length)
            {
                index = 0xFF & hysteresisBuf[ptr];
                histData[index]++;
                ptr++;
            }

            float sum = 0;

            for (int t = 0; t <= max; t++)
            {
                sum += t * histData[t];
            }

            float sumB = 0;
            int wB = 0;
            int wF = 0;

            float varMax = 0;
            int threshold = 0;

            int total = hysteresisBuf.Length;

            for (int t = 0; t <= max; t++)
            {
                // Weight background
                wB += histData[t];
                if (wB == 0) continue;

                // Weight foreground
                wF = total - wB;
                if (wF == 0) break;

                sumB += t * histData[t];

                // Mean Background
                float mB = sumB / wB;
                // Mean Foreground
                float mF = (sum - sumB) / wF;

                // Calculate "Between Class Variance"
                float varBetween = (float)wB * wF * (mB - mF) * (mB - mF);

                // Check if new maximum is found
                if (varBetween > varMax)
                {
                    varMax = varBetween;
                    threshold = t;
                }
            }

            MaxHysteresisThresh = threshold;
            MinHysteresisThresh = (int)(0.5 * threshold);
        }

        private void ApplyDoubleThresholding(int[,] nonMaxBuf, int kernelRadius)
        {
            edgePoints = new int[_imgWidth, _imgHeight];

            for (int x = 0 + kernelRadius; x < (_inspectArea.Width - kernelRadius); x++)
            {
                for (int y = 0 + kernelRadius; y < (_inspectArea.Height - kernelRadius); y++)
                {
                    // strong edge
                    if (nonMaxBuf[x, y] >= MaxHysteresisThresh)
                    {
                        edgePoints[x, y] = 1;
                    }

                    // weak edge
                    if ((nonMaxBuf[x, y] < MaxHysteresisThresh) && (nonMaxBuf[x, y] >= MinHysteresisThresh))
                    {
                        edgePoints[x, y] = 2;
                    }
                }
            }
        }

        private void TrackWeakEdge(int kernelRadius)
        {
            visitedMap = new bool[_imgWidth, _imgHeight];

            for (int x = 0 + kernelRadius; x < (_inspectArea.Width - kernelRadius); x++)
            {
                for (int y = 0 + kernelRadius; y < (_inspectArea.Height - kernelRadius); y++)
                {
                    // If strong edge
                    if (edgePoints[x, y] == 1)
                    {
                        EdgeMap[x, y] = 0xFF;
                        // And have not yet been visited, 
                        // find weak edge within the respective 8-connected neighborhood
                        if (!visitedMap[x, y])
                        {
                            TrackEdge(x, y);
                            visitedMap[x, y] = true;
                        }
                    }
                }
            }
        }

        private void TrackEdge(int x, int y)
        {
            //1
            if (edgePoints[x + 1, y] == 2)
            {
                edgePoints[x + 1, y] = 1;
                EdgeMap[x + 1, y] = 0xFF;
                visitedMap[x + 1, y] = true;
                TrackEdge(x + 1, y);
            }

            //2
            if (edgePoints[x + 1, y - 1] == 2)
            {
                edgePoints[x + 1, y - 1] = 1;
                EdgeMap[x + 1, y - 1] = 0xFF;
                visitedMap[x + 1, y - 1] = true;
                TrackEdge(x + 1, y - 1);
            }

            //3
            if (edgePoints[x, y - 1] == 2)
            {
                edgePoints[x, y - 1] = 1;
                EdgeMap[x, y - 1] = 0xFF;
                visitedMap[x, y - 1] = true;
                TrackEdge(x, y - 1);
            }

            //4
            if (edgePoints[x - 1, y - 1] == 2)
            {
                edgePoints[x - 1, y - 1] = 1;
                EdgeMap[x - 1, y - 1] = 0xFF;
                visitedMap[x - 1, y - 1] = true;
                TrackEdge(x - 1, y - 1);
            }

            //5
            if (edgePoints[x - 1, y] == 2)
            {
                edgePoints[x - 1, y] = 1;
                EdgeMap[x - 1, y] = 0xFF;
                visitedMap[x - 1, y] = true;
                TrackEdge(x - 1, y);
            }

            //6
            if (edgePoints[x - 1, y + 1] == 2)
            {
                edgePoints[x - 1, y + 1] = 1;
                EdgeMap[x - 1, y + 1] = 0xFF;
                visitedMap[x - 1, y + 1] = true;
                TrackEdge(x - 1, y + 1);
            }

            //7
            if (edgePoints[x, y + 1] == 2)
            {
                edgePoints[x, y + 1] = 1;
                EdgeMap[x, y + 1] = 0xFF;
                visitedMap[x, y + 1] = true;
                TrackEdge(x, y + 1);
            }

            //8
            if (edgePoints[x + 1, y + 1] == 2)
            {
                edgePoints[x + 1, y + 1] = 1;
                EdgeMap[x + 1, y + 1] = 0xFF;
                visitedMap[x + 1, y + 1] = true;
                TrackEdge(x + 1, y + 1);
            }
        }
    }
}