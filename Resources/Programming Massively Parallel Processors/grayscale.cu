#include <cuda_runtime.h>

const char* ROOT_DIR = "/Users/macintosh/UNM/GPU/Resources/Programming Massively Parallel Processors";
const char* input_path = ROOT_DIR "/flowers_color.ppm";
const char* output_path = ROOT_DIR "/flowers_rgb.pgm";

__constant__int CHANNELS;

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

// Minimal P6 reader: returns malloc'd RGB buffer (width*height*3) or nullptr on error.
// Assumes: exactly "P6", maxval 255, no comment lines.
static uint8_t* read_ppm_rgb_simple(const char* filename, int* width_out, int* height_out) {
    FILE* f = std::fopen(filename, "rb");
    if (!f) return nullptr;

    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1 || magic[0] != 'P' || magic[1] != '6') { std::fclose(f); return nullptr; }

    int width, height, maxval;
    if (fscanf(f, "%d %d %d", &width, &height, &maxval) != 3) { std::fclose(f); return nullptr; }
    if (maxval != 255) { std::fclose(f); return nullptr; }

    // consume single whitespace byte before binary data
    int c = fgetc(f);
    if (c == EOF) { std::fclose(f); return nullptr; }

    size_t bytes = (size_t)width * height * 3;
    uint8_t* buf = (uint8_t*)std::malloc(bytes);
    if (!buf) { std::fclose(f); return nullptr; }

    size_t read = std::fread(buf, 1, bytes, f);
    std::fclose(f);
    if (read != bytes) { std::free(buf); return nullptr; }

    if (width_out) *width_out = width;
    if (height_out) *height_out = height;
    return buf;
}

static void write_pgm_gray(const char* filename, const uint8_t* gray, int width, int height) {
    std::FILE* f = std::fopen(filename, "wb");
    if (!f) { std::perror("fopen"); std::exit(1); }
    std::fprintf(f, "P5\n%d %d\n255\n", width, height);
    std::fwrite(gray, 1, width * height, f);
    std::fclose(f);
}
// // ...existing code...

//     static void write_ppm_rgb(const char* filename, const uint8_t* rgba, int dim) {
//     // PPM P6 is RGB; we drop alpha.
//     std::FILE* f = std::fopen(filename, "wb");
//     if (!f) {
//         std::perror("fopen");
//         std::exit(1);
//     }

//     std::fprintf(f, "P6\n%d %d\n255\n", dim, dim);

//     for (int i = 0; i < dim * dim; i++) {
//         const uint8_t r = rgba[i * 4 + 0];
//         const uint8_t g = rgba[i * 4 + 1];
//         const uint8_t b = rgba[i * 4 + 2];
//         std::fputc(r, f);
//         std::fputc(g, f);
//         std::fputc(b, f);
//     }

//     std::fclose(f);
// }
void main()
{
    cudaMemcpyToSymbol(CHANNELS, 3, sizeof(int));
    int width, height;
    unsigned char* h_inputImage = read_ppm_rgb_simple(input_path, &width, &height);
    if (!h_inputImage) {
        printf("Failed to read input image\n");
        return;
    }

    
}