import numpy as np

def main():
    print("""# 7daf4d34-e3c1-4827-9777-a6793ffc3440
# 603fd3987364b09f9aacb70d1ed12c268e24dd56
# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2""")
    for i in np.arange(0, 90, 0.2):
        print(f"{i:.1f}; {i:.1f}; 0.0; 0.0; 0.0; 6.5; 0.0")

if __name__ == "__main__":
    main()