using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Morphology
{
    public class Dilation : MorphologyBase
    {
        private static byte GetMinValueFromBuffer(byte value, byte[] pixelBuf, int offset)
        {
            return Math.Min(value, pixelBuf[offset]);
        }

        /// <summary>
        /// Get dilated binary image
        /// </summary>
        /// <param name="srcImg">Source binary image</param>
        /// <param name="kernelSize">DIlation kernel size</param>
        /// <returns></returns>
        public static Task<Bitmap> GetDilationImage(Bitmap srcImg, int kernelSize)
        {
            return ApplyMorph(srcImg, kernelSize, 0xFF, GetMinValueFromBuffer);
        }
    }
}
