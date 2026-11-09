/**
 * High-performance chiral spin-network simulation (C++17).
 *
 * Compile: make
 * Run:     ./spin_network --N 100 --sweeps 100000 --J 1.2 --h 0.001 --kT 1.0
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

class ChiralSpinNetwork {
public:
    ChiralSpinNetwork(int N, double J, double h, double kT, unsigned int seed)
        : N_(N), J_(J), h_(h), kT_(kT), rng_(seed), lattice_(N * N) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double p_left = 0.5 + 0.5 * std::tanh(h / kT);
        for (auto& s : lattice_) {
            s = dist(rng_) < p_left ? 1 : -1;
        }
    }

    void sweep() {
        std::uniform_int_distribution<int> idx(0, N_ - 1);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int k = 0; k < N_ * N_; ++k) {
            int i = idx(rng_);
            int j = idx(rng_);
            double dE = energyDelta(i, j);
            if (dE <= 0.0 || dist(rng_) < std::exp(-dE / kT_)) {
                lattice_[i * N_ + j] *= -1;
            }
        }
    }

    double orderParameter() const {
        long long sum = 0;
        for (int s : lattice_) sum += s;
        return static_cast<double>(sum) / static_cast<double>(lattice_.size());
    }

    double binderCumulant() const {
        double m = orderParameter();
        double m2 = m * m;
        double m4 = m2 * m2;
        if (m2 < 1e-12) return 0.0;
        return 1.0 - m4 / (3.0 * m2 * m2);
    }

    void saveLattice(const std::string& filename) const {
        std::ofstream out(filename);
        for (int i = 0; i < N_; ++i) {
            for (int j = 0; j < N_; ++j) {
                out << lattice_[i * N_ + j] << (j + 1 == N_ ? '\n' : ' ');
            }
        }
    }

    void run(int nSweeps, int sampleInterval, int burnIn,
             std::vector<double>& sigmaSeries,
             std::vector<int>& times) {
        for (int sweep = 0; sweep < nSweeps; ++sweep) {
            sweepOnce();
            if (sweep >= burnIn && sweep % sampleInterval == 0) {
                sigmaSeries.push_back(orderParameter());
                times.push_back(sweep);
            }
        }
    }

private:
    int N_;
    double J_, h_, kT_;
    std::mt19937 rng_;
    std::vector<int8_t> lattice_;

    double energyDelta(int i, int j) const {
        int s = lattice_[i * N_ + j];
        int neighbors = lattice_[((i + 1) % N_) * N_ + j]
                      + lattice_[((i - 1 + N_) % N_) * N_ + j]
                      + lattice_[i * N_ + (j + 1) % N_]
                      + lattice_[i * N_ + (j - 1 + N_) % N_];
        return 2.0 * s * (J_ * neighbors + h_);
    }

    void sweepOnce() { sweep(); }
};

int main(int argc, char* argv[]) {
    int N = 100;
    int nSweeps = 100000;
    int sampleInterval = 100;
    double J = 1.2;
    double h = 1e-3;
    double kT = 1.0;
    unsigned int seed = 42;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--N" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (arg == "--sweeps" && i + 1 < argc) nSweeps = std::stoi(argv[++i]);
        else if (arg == "--interval" && i + 1 < argc) sampleInterval = std::stoi(argv[++i]);
        else if (arg == "--J" && i + 1 < argc) J = std::stod(argv[++i]);
        else if (arg == "--h" && i + 1 < argc) h = std::stod(argv[++i]);
        else if (arg == "--kT" && i + 1 < argc) kT = std::stod(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        else if (arg == "--help") {
            std::cout << "Usage: ./spin_network [options]\n"
                      << "  --N          lattice size (default 100)\n"
                      << "  --sweeps     total sweeps (default 100000)\n"
                      << "  --interval   sampling interval (default 100)\n"
                      << "  --J          coupling (default 1.2)\n"
                      << "  --h          external field (default 1e-3)\n"
                      << "  --kT         temperature (default 1.0)\n"
                      << "  --seed       random seed (default 42)\n";
            return 0;
        }
    }

    ChiralSpinNetwork model(N, J, h, kT, seed);
    std::vector<double> sigmaSeries;
    std::vector<int> times;

    int burnIn = nSweeps / 10;
    model.run(nSweeps, sampleInterval, burnIn, sigmaSeries, times);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "# sweep sigma\n";
    for (size_t i = 0; i < sigmaSeries.size(); ++i) {
        std::cout << times[i] << " " << sigmaSeries[i] << "\n";
    }

    std::cerr << "Final sigma: " << model.orderParameter() << "\n";
    std::cerr << "Binder cumulant: " << model.binderCumulant() << "\n";

    model.saveLattice("final_lattice.txt");

    return 0;
}
