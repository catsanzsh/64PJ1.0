// main.cpp
#include <SDL2/SDL.h>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <iostream>
#include <string>

// Neural Network Structure
class Neuron {
private:
    std::vector<float> weights;
    float bias;

public:
    Neuron(int inputs) : bias(0.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        weights.resize(inputs);
        for (auto& w : weights) {
            w = dis(gen);
        }
    }

    float activate(const std::vector<float>& inputs) {
        float sum = bias;
        for (size_t i = 0; i < inputs.size(); i++) {
            sum += inputs[i] * weights[i];
        }
        return tanh(sum); // Using tanh as activation function
    }
};

class NeuralNetwork {
private:
    std::vector<std::vector<Neuron>> layers;

public:
    NeuralNetwork(const std::vector<int>& topology) {
        for (size_t i = 1; i < topology.size(); i++) {
            std::vector<Neuron> layer;
            int numInputs = topology[i - 1];

            for (int j = 0; j < topology[i]; j++) {
                layer.emplace_back(numInputs);
            }
            layers.push_back(layer);
        }
    }

    std::vector<float> feedForward(std::vector<float> inputs) {
        for (const auto& layer : layers) {
            std::vector<float> outputs;
            for (const auto& neuron : layer) {
                outputs.push_back(neuron.activate(inputs));
            }
            inputs = outputs;
        }
        return inputs;
    }
};

// Game Unit
class Unit {
private:
    SDL_Rect rect;
    SDL_Texture* texture;
    NeuralNetwork brain;
    float x, y;
    float speed;

public:
    Unit(int startX, int startY, SDL_Renderer* renderer, const std::string& imagePath)
        : brain({4, 6, 2}), x(startX), y(startY), speed(2.0f) {
        rect = {static_cast<int>(x), static_cast<int>(y), 40, 40};

        // Load BMP image
        SDL_Surface* surface = SDL_LoadBMP(imagePath.c_str());
        if (!surface) {
            std::cerr << "Failed to load image: " << SDL_GetError() << std::endl;
            texture = nullptr;
        } else {
            texture = SDL_CreateTextureFromSurface(renderer, surface);
            SDL_FreeSurface(surface);
        }
    }

    ~Unit() {
        if (texture) {
            SDL_DestroyTexture(texture);
        }
    }

    void update(const SDL_Point& target) {
        // Calculate inputs for neural network
        float dx = target.x - x;
        float dy = target.y - y;
        float distance = sqrtf(dx * dx + dy * dy);
        float angle = atan2f(dy, dx);

        // Feed inputs to neural network
        std::vector<float> inputs = {
            distance / 800.0f,
            angle / static_cast<float>(M_PI),
            speed / 5.0f,
            0.0f // Placeholder for additional input
        };
        auto outputs = brain.feedForward(inputs);

        // Use outputs to control unit
        speed = outputs[0] * 5.0f;
        float direction = outputs[1] * static_cast<float>(M_PI);

        // Update position
        x += cosf(direction) * speed;
        y += sinf(direction) * speed;

        // Keep units within window bounds
        if (x < 0) x = 0;
        if (x > 800 - rect.w) x = 800 - rect.w;
        if (y < 0) y = 0;
        if (y > 600 - rect.h) y = 600 - rect.h;

        // Update rect position
        rect.x = static_cast<int>(x);
        rect.y = static_cast<int>(y);
    }

    void render(SDL_Renderer* renderer) {
        if (texture) {
            SDL_RenderCopy(renderer, texture, nullptr, &rect);
        } else {
            SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
            SDL_RenderFillRect(renderer, &rect);
        }
    }
};

// Game class
class Game {
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    bool running;
    std::vector<std::unique_ptr<Unit>> units;
    SDL_Point target;
    SDL_Texture* targetTexture;

public:
    Game() : window(nullptr), renderer(nullptr), running(false), targetTexture(nullptr) {
        target = {400, 300};
    }

    bool init() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
            return false;
        }

        // Enable Metal rendering for M1 Macs
        SDL_SetHint(SDL_HINT_RENDER_DRIVER, "metal");

        window = SDL_CreateWindow(
            "Neural Net Wars",
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            800, 600,
            SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI
        );

        if (!window) {
            std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
            SDL_Quit();
            return false;
        }

        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!renderer) {
            std::cerr << "Failed to create renderer: " << SDL_GetError() << std::endl;
            SDL_DestroyWindow(window);
            SDL_Quit();
            return false;
        }

        // Load target texture
        SDL_Surface* surface = SDL_LoadBMP("cat_target.bmp");
        if (!surface) {
            std::cerr << "Failed to load target image: " << SDL_GetError() << std::endl;
            // We can proceed without the target texture
            targetTexture = nullptr;
        } else {
            targetTexture = SDL_CreateTextureFromSurface(renderer, surface);
            SDL_FreeSurface(surface);
        }

        // Seed the random number generator
        std::srand(static_cast<unsigned int>(std::time(nullptr)));

        // Create initial units
        for (int i = 0; i < 5; i++) {
            units.push_back(std::make_unique<Unit>(
                rand() % 800, rand() % 600, renderer, "cat_unit.bmp"));
        }

        running = true;
        return true;
    }

    void handleEvents() {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                target.x = event.button.x;
                target.y = event.button.y;
            }
        }
    }

    void update() {
        for (auto& unit : units) {
            unit->update(target);
        }
    }

    void render() {
        // Clear screen
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black background
        SDL_RenderClear(renderer);

        // Render target
        if (targetTexture) {
            SDL_Rect targetRect = {target.x - 20, target.y - 20, 40, 40};
            SDL_RenderCopy(renderer, targetTexture, nullptr, &targetRect);
        } else {
            SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red dot
            SDL_Rect targetRect = {target.x - 5, target.y - 5, 10, 10};
            SDL_RenderFillRect(renderer, &targetRect);
        }

        // Render units
        for (auto& unit : units) {
            unit->render(renderer);
        }

        SDL_RenderPresent(renderer);
    }

    void clean() {
        if (targetTexture) {
            SDL_DestroyTexture(targetTexture);
        }
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    bool isRunning() const { return running; }
};

int main(int argc, char* argv[]) {
    Game game;

    if (!game.init()) {
        std::cout << "Failed to initialize game!" << std::endl;
        return -1;
    }

    while (game.isRunning()) {
        game.handleEvents();
        game.update();
        game.render();
        SDL_Delay(16); // Cap at ~60 FPS
    }

    game.clean();
    return 0;
}
