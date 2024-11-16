import subprocess
import time
from pathlib import Path
import json
from datetime import datetime

class HashcatTester:
    def __init__(self):
        # Use raw string for Windows path
        self.hashcat_path = r"SourceCode\hashcat-6.2.6\hashcat.exe"
        self.results_dir = Path(r"SourceCode/ReworldTest/Test_Results")
        self.md5_dir = Path(r"SourceCode/ReworldTest/MD5_Passwords")
        # Point to the actual dictionary file
        self.dict_path = Path(r"SourceCode\Main_Dict\Main_Dict_8.txt")
        # Add output directory for cracked passwords
        self.cracked_dir = Path(r"SourceCode/ReworldTest/Cracked_Passwords")
        self.max_time = 180  # 10 minutes in seconds
        
        # Change to hashcat directory before running (to fix OpenCL error)
        self.hashcat_dir = Path(self.hashcat_path).parent
        
        # Verify paths exist
        if not Path(self.hashcat_path).exists():
            raise FileNotFoundError(f"Hashcat not found at: {self.hashcat_path}")
        if not self.dict_path.exists():
            raise FileNotFoundError(f"Dictionary file not found at: {self.dict_path}")
        
        # Create necessary directories
        self.results_dir.mkdir(exist_ok=True)
        self.cracked_dir.mkdir(exist_ok=True)

    def run_attack(self, hash_file, attack_mode, dict_file=None):
        """Run hashcat attack and return results"""
        # Create more descriptive output filename
        attack_type = "dictionary" if attack_mode == 0 else "bruteforce"
        timestamp = int(time.time())
        outfile = (self.cracked_dir / f"{attack_type}_cracked_{hash_file.stem}_{timestamp}.txt").absolute()
        
        start_time = time.time()
        result = {
            'hash_file': hash_file.name,
            'attack_type': 'dictionary' if attack_mode == 0 else 'brute-force',
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'cracked_passwords': 0,
            'total_passwords': 0,
            'time_taken': 0,
            'success_rate': 0,
            'status': 'completed',
            'output_file': str(outfile)  # Store output file path in results
        }

        # Count total hashes in file
        with open(hash_file, 'r') as f:
            result['total_passwords'] = sum(1 for line in f)

        # Prepare crack command
        crack_cmd = [
            self.hashcat_path,
            '-m', '0',           # MD5 mode
            '-a', str(attack_mode),
            str(hash_file.absolute()),
            '--status',
            '--status-timer', '1',
            '-o', str(outfile),  # Output file for cracked passwords
            '-w', '3',           # Workload profile
            '-O',                # Optimize for 32 chars or less
            '--force'
        ]

        # Add attack-specific parameters
        if attack_mode == 0:  # Dictionary Attack
            crack_cmd.append(str(self.dict_path.absolute()))
        else:  # Brute Force Attack
            crack_cmd.extend([
                '--increment',
                '--increment-min', '8',
                '--increment-max', '8',
                '?a?a?a?a?a?a?a?a'
            ])

        print(f"\nExecuting crack command: {' '.join(crack_cmd)}")

        try:
            process = subprocess.Popen(
                crack_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(self.hashcat_dir)
            )

            # Monitor output in real-time
            try:
                while True:
                    if time.time() - start_time > self.max_time:
                        process.kill()
                        result['status'] = 'timeout'
                        break

                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        print(line.strip())

                stdout, stderr = process.communicate()
                if stdout:
                    print(stdout)
                if stderr:
                    print(f"Errors: {stderr}")

            except subprocess.TimeoutExpired:
                process.kill()
                result['status'] = 'timeout'

            # Wait a moment for files to be written
            time.sleep(1)

            # Count cracked passwords from specific output file
            if outfile.exists():
                with open(outfile, 'r') as f:
                    result['cracked_passwords'] = sum(1 for line in f if line.strip())
                    print(f"\nCracked passwords saved to: {outfile}")
                    print(f"Number of passwords cracked: {result['cracked_passwords']}")

        except Exception as e:
            print(f"Error: {str(e)}")
            result['status'] = f'error: {str(e)}'

        # Calculate final metrics
        result['time_taken'] = round(time.time() - start_time, 2)
        result['success_rate'] = round(
            (result['cracked_passwords'] / result['total_passwords']) * 100, 2
        ) if result['total_passwords'] > 0 else 0

        # Print result
        self._print_test_result(result)

        return result

    def _count_cracked_from_potfile(self, potfile):
        """Count cracked passwords from potfile"""
        try:
            if not potfile.exists():
                return 0
            with open(potfile, 'r') as f:
                return sum(1 for line in f if line.strip())
        except Exception as e:
            print(f"Error reading potfile: {e}")
            return 0

    def _print_test_result(self, result):
        """Print individual test result"""
        print("\n=== Test Result ===")
        print(f"File: {result['hash_file']}")
        print(f"Attack Type: {result['attack_type']}")
        print(f"Status: {result['status']}")
        print(f"Time Taken: {result['time_taken']} seconds")
        print(f"Cracked Passwords File: {result['output_file']}")
        print(f"Cracked: {result['cracked_passwords']}/{result['total_passwords']}")
        print(f"Success Rate: {result['success_rate']}%")

    def run_tests(self):
        """Run both dictionary and brute force tests on all hash files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = []

        # Process each MD5 file
        for hash_file in self.md5_dir.glob('*.txt'):
            print(f"\nTesting file: {hash_file.name}")
            
            # Dictionary Attack
            print("Running dictionary attack...")
            dict_result = self.run_attack(hash_file, 0, self.dict_path)  # Pass the dictionary file directly
            results.append(dict_result)
            
            # Brute Force Attack
            print("Running brute force attack...")
            brute_result = self.run_attack(hash_file, 3)
            results.append(brute_result)

        # Save results
        result_file = self.results_dir / f"test_results_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        self._print_summary(results)
        return result_file

    def _print_summary(self, results):
        """Print a summary of the test results"""
        print("\n=== Test Summary ===")
        for result in results:
            print(f"\nFile: {result['hash_file']}")
            print(f"Attack Type: {result['attack_type']}")
            print(f"Status: {result['status']}")
            print(f"Time Taken: {result['time_taken']} seconds")
            print(f"Cracked: {result['cracked_passwords']}/{result['total_passwords']}")
            print(f"Success Rate: {result['success_rate']}%")
            if result.get('last_progress'):
                print(f"Progress: {result['last_progress']}")

def main():
    try:
        tester = HashcatTester()
        result_file = tester.run_tests()
        print(f"\nDetailed results saved to: {result_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
