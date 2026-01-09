#include <stdio.h>
#include <stdlib.h>

#define ROOT_DIR "/usr/GPU/Resources/Programming Massively Parallel Processors"
const char* input_path = ROOT_DIR "/flowers_rgb.ppm";
const char* output_path = ROOT_DIR "/flowers_g.pgm";

static unsigned char* read_ppm_rgb_simple(const char* filename, int* width_out, int* height_out);
static void write_pgm_gray(const char* filename, const unsigned char* gray, int width, int height);

__constant__ int CHANNELS;

// The input image is encoded as unsigned chars [0, 255]
// Each pixel is 3 consecutive chars for the 3 channels (RGB)
__global__
void colorToGrayscaleConversionKernel(unsigned char* Pout,
    unsigned char* Pin, int width, int height) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (col < width && row < height) {
            // Get 1D offset for the grayscale image
            int grayscaleOffset = row * width + col;

            // One can think of the RGB image having CHANNEL
            // times more columns than the gray scale image
            int rgbOffset = grayscaleOffset * CHANNELS;
            unsigned char r = Pin[rgbOffset    ];   // Red value
            unsigned char g = Pin[rgbOffset + 1];   // Green value
            unsigned char b = Pin[rgbOffset + 2];   // Blue value
            
            // Perform the rescaling and store it
            // Multiply by floating point constants
            // * These weights are derived from color science experiments modeling human visual
            // sensitivity and how strongly each RGB channel contributes to perceived brightness.

            Pout[grayscaleOffset] = (0.21f * r) + (0.71f * g) + (0.07f * b); 
        }
    }

void colorToGrayscaleConversion(unsigned char* Pin_h, unsigned char* Pout_h, int width, int height) {
    int size = width * height * sizeof(unsigned char);
    unsigned char *Pin_d, *Pout_d;

    cudaMalloc((void **)&Pin_d, size);
    cudaMalloc((void **)&Pout_d, size);

    cudaMemcpy(Pin_d, Pin_h, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(32, 32);
    dim3 gridDim(ceil(width / blockDim.x), ceil(height / blockDim.y));

    // Launch ceil(width/32) x ceil(height/32) blocks of 32 x 32 threads each
    colorToGrayscaleConversionKernel<<<gridDim, blockDim>>>(Pout_d, Pin_d, width, height);

    
    cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost);
     
    cudaFree(Pin_d);
    cudaFree(Pout_d);
}

int main(int argc, char* argv[])
{
    int channels = 3;
    cudaMemcpyToSymbol(CHANNELS, &channels, sizeof(channels));
    int width, height;

    unsigned char* Pin = read_ppm_rgb_simple(input_path, &width, &height);
    if (!Pin) {
        printf("Failed to read input image %s\n", input_path);
        return 1;
    }

    printf("Image dimensions: %d x %d\n", width, height);

    int size = width * height * sizeof(unsigned char);
    unsigned char* Pout = (unsigned char*)malloc(size);

    colorToGrayscaleConversion(Pin, Pout, width, height);

    write_pgm_gray(output_path, Pout, width, height);

    free(Pin);
    free(Pout);
    
    return 0;


}

// Minimal P6 reader: returns malloc'd RGB buffer (width*height*3) or nullptr on error.
// Assumes: exactly "P6", maxval 255, no comment lines.
static unsigned char* read_ppm_rgb_simple(const char* filename, int* width_out, int* height_out) {
    FILE* f = fopen(filename, "rb");
    if (!f) return nullptr;

    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1 || magic[0] != 'P' || magic[1] != '6') { fclose(f); return nullptr; }
    
    int width, height, maxval;
    if (fscanf(f, "%d %d %d", &width, &height, &maxval) != 3) { fclose(f); return nullptr; }
    if (maxval != 255) { fclose(f); return nullptr; }


    // consume single whitespace byte before binary data
    int c = fgetc(f);
    if (c == EOF) { fclose(f); return nullptr; }
    size_t bytes = (size_t)width * height * 3;
    unsigned char* buf = (unsigned char*)malloc(bytes);
    if (!buf) { fclose(f); return nullptr; }

    size_t read = fread(buf, 1, bytes, f);
    fclose(f);
    if (read != bytes) { free(buf); return nullptr; }

    if (width_out) *width_out = width;
    if (height_out) *height_out = height;
    return buf;
}

static void write_pgm_gray(const char* filename, const unsigned char* gray, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) { perror("fopen"); exit(1); }
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(gray, 1, width * height, f);
    fclose(f);
}