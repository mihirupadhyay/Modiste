import os
import subprocess
import pkg_resources

def get_available_versions(package_name):
    """Get available versions of a package from PyPI."""
    try:
        cmd = f"pip index versions {package_name}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        versions = [line.split()[-1] for line in result.stdout.split('\n') if 'Available versions:' in line]
        return versions[:5]  # Return the first 5 versions
    except Exception:
        return []

def check_install(package, env_name):
    package_name = package.split('==')[0]
    failed_packages = []

    # Activate the environment and check for the package
    conda_check_cmd = f"conda list -n {env_name} {package_name}"
    result = subprocess.run(conda_check_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if package_name not in result.stdout.decode():
        print(f"{package} not found. Attempting to install.")
        
        installers = [
            ("conda", f"conda install -n {env_name} {package} -y"),
            ("pip", f"pip install {package}"),
            ("easy_install", f"easy_install {package}")
        ]

        for installer, cmd in installers:
            print(f"Trying to install {package} with {installer}...")
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode == 0:
                print(f"Successfully installed {package} with {installer}.")
                return []

        print(f"Failed to install {package} with all methods. Trying alternative versions...")
        
        versions = get_available_versions(package_name)
        for version in versions:
            alt_package = f"{package_name}=={version}"
            print(f"Attempting to install {alt_package}...")
            
            for installer, cmd in installers:
                cmd = cmd.replace(package, alt_package)
                result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if result.returncode == 0:
                    print(f"Successfully installed {alt_package} with {installer}.")
                    return []

        print(f"Failed to install {package} and its alternative versions.")
        failed_packages.append(package)
    else:
        print(f"{package} is already installed.")

    return failed_packages

# Read the requirements file and check/install each package
failed_installations = []
with open("requirements_all.txt") as file:
    packages = file.readlines()

for package in packages:
    package = package.strip()
    if package:
        failed = check_install(package, "modiste")
        failed_installations.extend(failed)

if failed_installations:
    print("\nThe following packages failed to install:")
    for pkg in failed_installations:
        print(f"- {pkg}")
else:
    print("\nAll packages were successfully installed.")