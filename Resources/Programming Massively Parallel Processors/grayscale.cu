#include <cuda_runtime.h>

__constant__ int CHANNELS;

// The input image is encoded as unsigned chars [0, 255]
// Each pixel is 3 consecutive chars for the 3 channels (RGB)
__global__
void colorToGrayscaleConversion(unsigned char* Pout,
    unsigned char* Pin, int width, int height) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (col < width && row < height) {
            int grayscaleOffset = row * width + col;
            int rgbOffset = grayscaleOffset * CHANNELS;
            unsigned char r = Pin[rgbOffset    ];   // Red value
            unsigned char g = Pin[rgbOffset + 1];   // Green value
            unsigned char b = Pin[rgbOffset + 2];   // Blue value
            
            // Perform the rescaling and store it
            // Multiply by floating point constants
            Pout[grayscaleOffset] = (0.21f * r) + (0.71f * g) + (0.07 * b); 
        }
    }

void main()
{
    cudaMemcpyToSymbol(CHANNELS, 3, sizeof(int));
    
}