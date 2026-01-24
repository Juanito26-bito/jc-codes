"""
Rejection Sampling for Planck's Law
====================================
This script implements the accept-reject sampling method to generate random
numbers according to Planck's law for spectral emissive power.
"""

import numpy as np
import yaml
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import sys
import os


class PlanckLawSampler:
    """
    Implements rejection sampling for Planck's law distribution.
    
    Planck's Law: M(λ,T) = (2πhc²)/λ⁵ * 1/(exp(hc/(λk_B T)) - 1)
    """
    
    def __init__(self, config_file: str):
        """
        Initialize the sampler with configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.config = self._load_config(config_file)
        self._extract_parameters()
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_file}' not found.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            sys.exit(1)
    
    def _extract_parameters(self):
        """Extract parameters from configuration."""
        # Physical constants
        self.h = self.config['constants']['h']
        self.c = self.config['constants']['c']
        self.k_B = self.config['constants']['k_B']
        
        # Temperature
        self.T = self.config['temperature']['T']
        
        # Wavelength range
        self.lambda_min = self.config['wavelength']['lambda_min']
        self.lambda_max = self.config['wavelength']['lambda_max']
        
        # Sampling parameters
        self.n_samples = self.config['sampling']['n_samples']
        self.max_iterations = self.config['sampling']['max_iterations']
        
        # Compute maximum value of Planck function for normalization
        self.M_max = self._find_maximum()
        
        print(f"\nParameters loaded:")
        print(f"  Temperature: {self.T} K")
        print(f"  Wavelength range: [{self.lambda_min*1e9:.1f}, {self.lambda_max*1e9:.1f}] nm")
        print(f"  Number of samples: {self.n_samples}")
        print(f"  Maximum M(λ,T): {self.M_max:.2e}\n")
    
    def planck_law(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Calculate Planck's law for spectral emissive power.
        
        Args:
            wavelength: Wavelength(s) in meters
            
        Returns:
            Spectral emissive power M(λ,T)
        """
        # Avoid division by zero and overflow
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            numerator = 2 * np.pi * self.h * self.c**2
            denominator = wavelength**5 * (
                np.exp(self.h * self.c / (wavelength * self.k_B * self.T)) - 1
            )
            M = numerator / denominator
            
        # Handle any infinities or NaNs
        M = np.where(np.isfinite(M), M, 0)
        return M
    
    def _find_maximum(self) -> float:
        """
        Find the maximum value of Planck's law in the given wavelength range.
        This is used to normalize the distribution for rejection sampling.
        
        Returns:
            Maximum value of M(λ,T)
        """
        # Wien's displacement law gives the peak wavelength
        lambda_peak = 2.897771955e-3 / self.T  # Wien's displacement constant
        
        # Evaluate at peak and boundaries
        test_wavelengths = np.array([
            self.lambda_min,
            lambda_peak,
            self.lambda_max
        ])
        
        # Also sample finely around the peak
        if self.lambda_min < lambda_peak < self.lambda_max:
            fine_range = np.linspace(
                lambda_peak * 0.5, 
                lambda_peak * 1.5, 
                1000
            )
            test_wavelengths = np.concatenate([test_wavelengths, fine_range])
        
        # Filter to valid range
        test_wavelengths = test_wavelengths[
            (test_wavelengths >= self.lambda_min) & 
            (test_wavelengths <= self.lambda_max)
        ]
        
        values = self.planck_law(test_wavelengths)
        max_value = np.max(values)
        
        # Add a small margin to ensure all values are below M_max
        return max_value * 1.1
    
    def rejection_sampling(self) -> Tuple[np.ndarray, int, float]:
        """
        Perform rejection sampling to generate samples from Planck's distribution.
        
        Returns:
            samples: Array of accepted wavelength samples
            total_iterations: Total number of iterations performed
            acceptance_rate: Fraction of accepted samples
        """
        samples = []
        iterations = 0
        
        print("Starting rejection sampling...")
        
        while len(samples) < self.n_samples and iterations < self.max_iterations:
            # Generate candidate from uniform proposal distribution
            lambda_candidate = np.random.uniform(self.lambda_min, self.lambda_max)
            
            # Generate uniform random number for acceptance test
            u = np.random.uniform(0, self.M_max)
            
            # Calculate Planck's law at candidate wavelength
            M_candidate = self.planck_law(lambda_candidate)
            
            # Accept or reject
            if u <= M_candidate:
                samples.append(lambda_candidate)
            
            iterations += 1
            
            # Progress update
            if iterations % 10000 == 0:
                acceptance_rate = len(samples) / iterations
                print(f"  Iterations: {iterations}, Samples: {len(samples)}, "
                      f"Acceptance rate: {acceptance_rate:.4f}")
        
        if iterations >= self.max_iterations:
            print(f"\nWarning: Reached maximum iterations ({self.max_iterations})")
            print(f"Only generated {len(samples)} samples out of {self.n_samples} requested")
        
        samples = np.array(samples)
        acceptance_rate = len(samples) / iterations
        
        print(f"\nSampling complete!")
        print(f"  Total iterations: {iterations}")
        print(f"  Samples generated: {len(samples)}")
        print(f"  Final acceptance rate: {acceptance_rate:.4f}")
        
        return samples, iterations, acceptance_rate
    
    def plot_results(self, samples: np.ndarray):
        """
        Plot the histogram of samples against the theoretical Planck distribution.
        
        Args:
            samples: Array of sampled wavelengths
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Convert wavelengths to nanometers for plotting
        samples_nm = samples * 1e9
        lambda_min_nm = self.lambda_min * 1e9
        lambda_max_nm = self.lambda_max * 1e9
        
        # Plot 1: Histogram vs Theoretical Distribution
        ax1.hist(samples_nm, bins=50, density=True, alpha=0.6, 
                color='blue', edgecolor='black', label='Sampled distribution')
        
        # Plot theoretical Planck curve
        lambda_theory = np.linspace(self.lambda_min, self.lambda_max, 1000)
        M_theory = self.planck_law(lambda_theory)
        
        # Normalize to match histogram
        lambda_range = self.lambda_max - self.lambda_min
        M_theory_normalized = M_theory / (np.trapz(M_theory, lambda_theory) / lambda_range)
        
        ax1.plot(lambda_theory * 1e9, M_theory_normalized, 'r-', 
                linewidth=2, label='Theoretical Planck distribution')
        
        ax1.set_xlabel('Wavelength (nm)', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title(f'Rejection Sampling of Planck\'s Law (T = {self.T} K)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Planck's Law (non-normalized)
        ax2.plot(lambda_theory * 1e9, M_theory, 'g-', linewidth=2)
        ax2.set_xlabel('Wavelength (nm)', fontsize=12)
        ax2.set_ylabel('Spectral Emissive Power (W/m²/m)', fontsize=12)
        ax2.set_title('Planck\'s Law - Spectral Emissive Power', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        # Save in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(script_dir, 'planck_sampling_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to planck_sampling_results.png")
        plt.show()
    
    def save_samples(self, samples: np.ndarray):
        """
        Save samples to CSV file.
        
        Args:
            samples: Array of sampled wavelengths
        """
        if self.config['output']['save_samples']:
            output_file = self.config['output']['output_file']
            # Save in the same directory as the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, output_file)
            np.savetxt(output_path, samples, 
                      header='Wavelength (m)', comments='', fmt='%.10e')
            print(f"Samples saved to {output_file}")
    
    def run(self):
        """Run the complete sampling pipeline."""
        # Perform rejection sampling
        samples, iterations, acceptance_rate = self.rejection_sampling()
        
        # Generate statistics
        print(f"\nSample Statistics:")
        print(f"  Mean wavelength: {np.mean(samples)*1e9:.2f} nm")
        print(f"  Median wavelength: {np.median(samples)*1e9:.2f} nm")
        print(f"  Std deviation: {np.std(samples)*1e9:.2f} nm")
        print(f"  Min wavelength: {np.min(samples)*1e9:.2f} nm")
        print(f"  Max wavelength: {np.max(samples)*1e9:.2f} nm")
        
        # Save samples
        self.save_samples(samples)
        
        # Plot results if requested
        if self.config['output']['plot']:
            self.plot_results(samples)
        
        return samples


def main():
    """Main function to run the rejection sampling."""
    # Configuration file path - always in same directory as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, 'config.yml')
    
    # Create sampler and run
    sampler = PlanckLawSampler(config_file)
    samples = sampler.run()
    
    print("\n" + "="*60)
    print("Rejection Sampling Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
