using System;
using System.Drawing;
using System.Threading.Tasks;

namespace Morphology
{
    public class Erosion : MorphologyBase
    {
        private static byte GetMaxValueFromBuffer(byte value, byte[] pixelBuf, int offset)
        {
            return Math.Max(value, pixelBuf[offset]);
        }

        /// <summary>
        /// Get eroded binary image
        /// </summary>
        /// <param name="srcImg">Source binary image</param>
        /// <param name="kernelSize">Erosion kernel size</param>
        /// <returns></returns>
        public static Task<Bitmap> GetErosionImage(Bitmap srcImg, int kernelSize)
        {
            return ApplyMorph(srcImg, kernelSize, 0x00, GetMaxValueFromBuffer);
        }
    }
}
