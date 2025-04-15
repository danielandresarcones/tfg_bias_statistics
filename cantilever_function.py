import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CantileverBeam:
    def __init__(self, length, height, young_modulus, yield_stress, softening_factor):
        self.length = length  # Beam length (m)
        self.height = height  # Beam height (m)
        self.initial_young_modulus = young_modulus  # Initial Young's modulus (Pa)
        self.current_young_modulus = young_modulus  # Current Young's modulus after softening (Pa)
        self.yield_stress = yield_stress  # Yield stress (Pa)
        self.softening_factor = softening_factor  # Softening factor (dimensionless)

    def moment_of_inertia(self):
        """Calculate the moment of inertia for a rectangular cross-section."""
        return (self.height ** 3) / 12

    def stress(self, load):
        """Calculate the bending stress at the fixed end of the cantilever beam."""
        I = self.moment_of_inertia()
        max_stress = (load * self.length * self.height / 2) / I
        return max_stress

    def deflection(self, load):
        """Calculate the tip deflection for the given load using the current Young's modulus."""
        I = self.moment_of_inertia()
        delta = (load * self.length ** 3) / (3 * self.current_young_modulus * I)
        return delta

    def generate_data(self, loads, noise_std=0.0):
        """Generate a dataset of load, stress, deflection, and softened stress."""
        data = []
        for load in loads:
            stress = self.stress(load)
            deflection = self.deflection(load)
            softened_stress = stress

            # Apply strain softening if the stress exceeds the yield stress
            if stress > self.yield_stress:
                softened_stress = self.softening_factor * (stress - self.yield_stress)
                # Permanently reduce Young's modulus after softening
                self.current_young_modulus *= (1-(1-self.softening_factor)*(stress - self.yield_stress)/self.yield_stress)

            # Add noise to the deflection
            if noise_std > 0:
                deflection += np.random.normal(0, noise_std)

            data.append((load, stress, deflection, softened_stress, self.current_young_modulus))

        df = pd.DataFrame(data, columns=["Load", "Stress", "Deflection", "Softened Stress", "Young's Modulus"])
        return df

    @staticmethod
    def plot_load_displacement(data, output_path=None):
        """Plot the load-displacement curve."""
        
        # Assuming df is your DataFrame
        plt.figure(figsize=(8, 6))

        # Scatter plot of Load vs Deflection
        scatter = plt.scatter(data['Load'], data['Deflection'], c=data["Young's Modulus"], cmap='viridis', s=50)

        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Young's Modulus")

        # Set labels and title
        plt.xlabel("Load")
        plt.ylabel("Deflection")
        plt.title("Load vs Deflection with Color by Young's Modulus")

        # Show the plot
        plt.savefig(output_path) if output_path else plt.show()
