#include <stdio.h>
#include <stdlib.h>

#define ROOT_DIR "/usr/GPU/Resources/Programming Massively Parallel Processors"
const char* input_path = ROOT_DIR "/flowers_rgb.ppm";
const char* output_path = ROOT_DIR "/flowers_blur.ppm";

static unsigned char* read_ppm_rgb(const char* filename, int* width_out, int* height_out);
static void write_pgm_gray(const char* filename, const unsigned char* gray, int width, int height);
static void write_ppm_rgb(const char* filename, const unsigned char* rgb, int width, int height);

// The value of BLURE_SIZE gives the number of pixels on each side (radius) of
// the patch and 2*BLURE_SIZE+1 gives the total number of pixels across one dimension
// of the patch. For example, for a 3x3 patch, BLUR_SIZE = is set to 1, whereas for a
// 7x7 patch, BLUR_SIZE is set to 3.
__constant__ int BLUR_SIZE;

// __global__
// void blurKernel(unsigned char* in,
//     unsigned char* out, int w, int h) {
//         int row = blockIdx.y*blockDim.y + threadIdx.y;
//         int col = blockIdx.x*blockDim.x + threadIdx.x;

//         if (col < w && row < h) {
//             int pixVal = 0;
//             int pixels = 0;

//             // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
//             for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
//                 for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
//                     int curRow = row + blurRow;
//                     int curCol = col + blurCol;

//                     // Verify we have a valid image pixel
//                     if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
//                         pixVal += in[curRow*w + curCol];
//                         ++pixels; // Keep track of number of pixels in the avg
//                     }
//                 }
//             }

//             // Write our new pixel value out
//             out[row*w + col] = (unsigned char)(pixVal/pixels);
//         }
//     }
__global__
void blurKernel(const unsigned char* in,
    unsigned char* out, int w, int h, int channels) {
        const int BLUR_SIZE = 3;

        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;

        if (col < w && row < h) {
            int pixels = 0;
            int sum[3] = {0, 0, 0};

            // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
            for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
                for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    // Verify we have a valid image pixel
                    if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                        int base = (curRow * w + curCol) * channels;
                        for (int c = 0; c < channels; ++c)
                            sum[c] += in[base + c];
                        ++pixels;
                    }
                }
            }

            // Write our new pixel value out (for each channel)
            int outBase = (row * w + col) * channels;
            for (int c = 0; c < channels; ++c)
                out[outBase + c] = (unsigned char)(sum[c] / pixels);
        }
    }
// void blur(unsigned char* Pin_h, unsigned char* Pout_h, int width, int height) {
//     int size = width * height * sizeof(unsigned char);
//     unsigned char *Pin_d, *Pout_d;

//     cudaMalloc((void **)&Pin_d, size);
//     cudaMalloc((void **)&Pout_d, size);

//     cudaMemcpy(Pin_d, Pin_h, size, cudaMemcpyHostToDevice);
    
//     dim3 blockDim(32, 32);
//     dim3 gridDim(ceil(width / blockDim.x), ceil(height / blockDim.y));

//     // Launch ceil(width/32) x ceil(height/32) blocks of 32 x 32 threads each
//     blurKernel<<<gridDim, blockDim>>>(Pout_d, Pin_d, width, height);

//     cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost);
     
//     cudaFree(Pin_d);
//     cudaFree(Pout_d);
// }
void blur(unsigned char* Pin_h, unsigned char* Pout_h, int width, int height) {
    const int channels = 3;
    size_t size = (size_t)width * height * channels * sizeof(unsigned char);
    unsigned char *Pin_d = nullptr, *Pout_d = nullptr;

    cudaMalloc((void **)&Pin_d, size);
    cudaMalloc((void **)&Pout_d, size);

    cudaMemcpy(Pin_d, Pin_h, size, cudaMemcpyHostToDevice);

    // set blur radius (e.g., 1 -> 3x3 kernel)
    int hostBlurSize = 1;
    cudaMemcpyToSymbol(BLUR_SIZE, &hostBlurSize, sizeof(int));

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Launch blocks
    blurKernel<<<gridDim, blockDim>>>(Pin_d, Pout_d, width, height, channels);
    cudaDeviceSynchronize(); // catch errors

    cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost);
     
    cudaFree(Pin_d);
    cudaFree(Pout_d);
}

int main(int argc, char* argv[])
{
    int blur_size = 3;
    cudaMemcpyToSymbol(BLUR_SIZE, &blur_size, sizeof(blur_size));
    printf("BLUR_SIZE=%d\n", BLUR_SIZE);

    int width, height;

    unsigned char* Pin = read_ppm_rgb(input_path, &width, &height);
    if (!Pin) {
        printf("Failed to read input image %s\n", input_path);
        return 1;
    }
    printf("Read file %s\n", input_path);

    int size = width * height * 3 * sizeof(unsigned char);
    unsigned char* Pout = (unsigned char*)malloc(size);

    blur(Pin, Pout, width, height);

    write_ppm_rgb(output_path, Pout, width, height);
    printf("Wrote file to %s\n", output_path);


    free(Pin);
    free(Pout);
    
    return 0;


}

// Minimal P6 reader: returns malloc'd RGB buffer (width*height*3) or nullptr on error.
// Assumes: exactly "P6", maxval 255, no comment lines.
static unsigned char* read_ppm_rgb(const char* filename, int* width_out, int* height_out) {
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

static void write_ppm_rgb(const char* filename, const unsigned char* rgb, int width, int height) {
    // PPM P6 is RGB; we drop alpha.
    FILE* f = fopen(filename, "wb");
    if (!f) {
        perror("fopen");
        exit(1);
    }

    fprintf(f, "P6\n%d %d\n255\n", width, height);

    for (int i = 0; i < width * height; ++i) {
        const unsigned char r = rgb[i * 3 + 0];
        const unsigned char g = rgb[i * 3 + 1];
        const unsigned char b = rgb[i * 3 + 2];
        
        
        fputc(r, f);
        fputc(g, f);
        fputc(b, f);
    }

    fclose(f);
}

