using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Morphology
{
    public abstract class MorphologyBase
    {
        private static byte[,] MorphKernel
        {
            get
            {
                return new byte[,]
                {
                    { 0, 1, 0 },
                    { 1, 1, 1 },
                    { 0, 1, 0 }
                };
            }
        }        

        protected static Task<Bitmap> ApplyMorph(Bitmap srcImg, int kernelSize, byte pixelValue, Func<byte, byte[], int, byte> f)
        {
            return Task.Run(() =>
            {
                int width = srcImg.Width;
                int height = srcImg.Height;

                //ImageData data = BitmapProcessing.GetBitmapData(srcImg);
                Bitmap tempBmp = (Bitmap)srcImg.Clone();
                Rectangle rect = new Rectangle(0, 0, tempBmp.Width, tempBmp.Height);
                BitmapData bmpData = tempBmp.LockBits(rect, ImageLockMode.ReadOnly, srcImg.PixelFormat);

                // Stride = length of each scan line
                int stride = bmpData.Stride;

                //Create byte arrays that will hold pixels data
                byte[] pixelBuffer = new byte[stride * tempBmp.Height];

                //Write pixel data to array meant for processing
                Marshal.Copy(bmpData.Scan0, pixelBuffer, 0, pixelBuffer.Length);

                tempBmp.UnlockBits(bmpData);

                byte[] resultBuffer = new byte[stride * height];

                int kernelDim = kernelSize;

                //This is the offset of center pixel from border of the kernel
                int kernelOffset = (kernelDim - 1) / 2;
                int calcOffset = 0;
                int byteOffset = 0;
                byte value = 0x00;

                for (int y = kernelOffset; y < height - kernelOffset; y++)
                {
                    for (int x = kernelOffset; x < width - kernelOffset; x++)
                    {
                        value = pixelValue;
                        byteOffset = y * stride + x;

                        for (int ykernel = -kernelOffset; ykernel <= kernelOffset; ykernel++)
                        {
                            for (int xkernel = -kernelOffset; xkernel <= kernelOffset; xkernel++)
                            {
                                if (MorphKernel[ykernel + kernelOffset, xkernel + kernelOffset] == 1)
                                {
                                    calcOffset = byteOffset + ykernel * stride + xkernel;
                                    value = f(value, pixelBuffer, calcOffset);
                                }
                                else
                                {
                                    continue;
                                }
                            }
                        }

                        //Write processed data into resultBuffer
                        resultBuffer[byteOffset] = value;
                    }
                }

                // Create a new Bitmap
                Bitmap binImage = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
                // Specify the portion of the Bitmap to lock.
                Rectangle canvas = new Rectangle(0, 0, binImage.Width, binImage.Height);
                // Lock binImage in system memory so that it can be changed programmatically.
                // The BitmapData specifies the attributes of the Bitmap, such as size, pixel format,
                // the starting address of the pixel data in memory, and length of each scan line (stride).
                BitmapData binImageData = binImage.LockBits(canvas, ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);

                // Copy data to binImageData.
                Marshal.Copy(resultBuffer, 0, binImageData.Scan0, resultBuffer.Length);
                // Unlock the bits.
                binImage.UnlockBits(binImageData);

                return binImage;
            });
        }
    }
}
