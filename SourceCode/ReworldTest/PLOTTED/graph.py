import json
import pandas as pd
import matplotlib.pyplot as plt

# Complete JSON data
data = [
    {"hash_file": "md5s_lowercase_8.txt", "attack_type": "dictionary", "start_time": "2024-11-11 03:45:52", "cracked_passwords": 468, "total_passwords": 10000, "time_taken": 46.8, "success_rate": 2.6, "status": "completed"},
    {"hash_file": "md5s_lowercase_8.txt", "attack_type": "brute-force", "start_time": "2024-11-11 03:48:29", "cracked_passwords": 300, "total_passwords": 10000, "time_taken": 30.0, "success_rate": 1.4, "status": "timeout"},
    {"hash_file": "md5s_lowercase_numbers_8.txt", "attack_type": "dictionary", "start_time": "2024-11-11 03:51:32", "cracked_passwords": 65, "total_passwords": 10000, "time_taken": 143.38, "success_rate": 1.8, "status": "completed"},
    {"hash_file": "md5s_lowercase_numbers_8.txt", "attack_type": "brute-force", "start_time": "2024-11-11 03:53:56", "cracked_passwords": 80, "total_passwords": 10000, "time_taken": 183.72, "success_rate": 0.8, "status": "timeout"},
    {"hash_file": "md5s_lowercase_symbols_8.txt", "attack_type": "dictionary", "start_time": "2024-11-11 03:57:00", "cracked_passwords": 2, "total_passwords": 10000, "time_taken": 141.55, "success_rate": 0.2, "status": "completed"},
    {"hash_file": "md5s_lowercase_symbols_8.txt", "attack_type": "brute-force", "start_time": "2024-11-11 03:59:21", "cracked_passwords": 8, "total_passwords": 10000, "time_taken": 182.82, "success_rate": 0.8, "status": "timeout"},
    {"hash_file": "md5s_numbers_8.txt", "attack_type": "dictionary", "start_time": "2024-11-11 04:02:24", "cracked_passwords": 60, "total_passwords": 10000, "time_taken": 142.35, "success_rate": 6.0, "status": "completed"},
    {"hash_file": "md5s_numbers_8.txt", "attack_type": "brute-force", "start_time": "2024-11-11 04:03:28", "cracked_passwords": 499, "total_passwords": 10000, "time_taken": 182.76, "success_rate": 49.9, "status": "timeout"},
    {"hash_file": "md5s_symbols_8.txt", "attack_type": "dictionary", "start_time": "2024-11-11 04:06:31", "cracked_passwords": 00, "total_passwords": 10000, "time_taken": 141.43, "success_rate": 0, "status": "completed"},
    {"hash_file": "md5s_symbols_8.txt", "attack_type": "brute-force", "start_time": "2024-11-11 04:08:53", "cracked_passwords": 00, "total_passwords": 10000, "time_taken": 182.76, "success_rate": 0, "status": "timeout"},
    {"hash_file": "md5s_uppercase_8.txt", "attack_type": "dictionary", "start_time": "2024-11-11 04:11:55", "cracked_passwords": 250, "total_passwords": 10000, "time_taken": 142.46, "success_rate": 25.0, "status": "completed"},
    {"hash_file": "md5s_uppercase_8.txt", "attack_type": "brute-force", "start_time": "2024-11-11 04:14:18", "cracked_passwords": 550, "total_passwords": 10000, "time_taken": 182.86, "success_rate": 55.0, "status": "timeout"},
    {"hash_file": "md5s_uppercase_lowercase_8.txt", "attack_type": "dictionary", "start_time": "2024-11-11 04:17:21", "cracked_passwords": 350, "total_passwords": 10000, "time_taken": 200.45, "success_rate": 30, "status": "completed"},
    {"hash_file": "md5s_uppercase_lowercase_8.txt", "attack_type": "brute-force", "start_time": "2024-11-11 04:19:42", "cracked_passwords": 310, "total_passwords": 10000, "time_taken": 325.69, "success_rate": 27.9, "status": "timeout"},
    {"hash_file": "md5s_uppercase_lowercase_numbers_8.txt", "attack_type": "dictionary", "start_time": "2024-11-11 04:22:45", "cracked_passwords": 650, "total_passwords": 10000, "time_taken": 250.5, "success_rate": 55.0, "status": "completed"},
    {"hash_file": "md5s_uppercase_lowercase_numbers_8.txt", "attack_type": "brute-force", "start_time": "2024-11-11 04:25:08", "cracked_passwords": 504, "total_passwords": 10000, "time_taken": 183.7, "success_rate": 65.0, "status": "timeout"},
    {"hash_file": "md5s_uppercase_lowercase_symbols_8.txt", "attack_type": "dictionary", "start_time": "2024-11-11 04:28:12", "cracked_passwords": 2, "total_passwords": 10000, "time_taken": 143.06, "success_rate": 0.2, "status": "completed"},
    {"hash_file": "md5s_uppercase_lowercase_symbols_8.txt", "attack_type": "brute-force", "start_time": "2024-11-11 04:30:35", "cracked_passwords": 24, "total_passwords": 10000, "time_taken": 183.72, "success_rate": 2.4, "status": "timeout"},
    {"hash_file": "md5s_uppercase_numbers_8.txt", "attack_type": "dictionary", "start_time": "2024-11-11 04:33:38", "cracked_passwords": 285, "total_passwords": 10000, "time_taken": 163.74, "success_rate": 28.5, "status": "completed"},
    {"hash_file": "md5s_uppercase_numbers_8.txt", "attack_type": "brute-force", "start_time": "2024-11-11 04:36:00", "cracked_passwords": 320, "total_passwords": 10000, "time_taken": 183.7, "success_rate": 32.0, "status": "timeout"},
    {"hash_file": "md5s_uppercase_number_symbols_8.txt", "attack_type": "dictionary", "start_time": "2024-11-11 04:39:04", "cracked_passwords": 35, "total_passwords": 10000, "time_taken": 195.6, "success_rate": 3.5, "status": "completed"},
    {"hash_file": "md5s_uppercase_number_symbols_8.txt", "attack_type": "brute-force", "start_time": "2024-11-11 04:41:25", "cracked_passwords": 56, "total_passwords": 10000, "time_taken": 183.64, "success_rate": 5.6, "status": "timeout"},
    {"hash_file": "md5s_uppercase_symbols_8.txt", "attack_type": "dictionary", "start_time": "2024-11-11 04:44:29", "cracked_passwords": 154, "total_passwords": 10000, "time_taken": 144.06, "success_rate": 15.4, "status": "completed"},
    {"hash_file": "md5s_uppercase_symbols_8.txt", "attack_type": "brute-force", "start_time": "2024-11-11 04:46:53", "cracked_passwords": 200, "total_passwords": 10000, "time_taken": 183.69, "success_rate": 20.0, "status": "timeout"}
]

# Load data into DataFrame
df = pd.DataFrame(data)

# Convert start_time to datetime for easier plotting and sorting
df['start_time'] = pd.to_datetime(df['start_time'])

# Plot Success Rate Comparison: Dictionary vs Brute-force
plt.figure(figsize=(12, 6))
for attack_type in ['dictionary', 'brute-force']:
    subset = df[df['attack_type'] == attack_type]
    plt.plot(subset['hash_file'], subset['success_rate'], marker='o', label=f'{attack_type.capitalize()} Attack')

plt.xlabel('Hash File')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate Comparison by Attack Type')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Plot Time Taken Comparison for Dictionary and Brute-force Attacks
plt.figure(figsize=(12, 6))
for attack_type in ['dictionary', 'brute-force']:
    subset = df[df['attack_type'] == attack_type]
    plt.plot(subset['hash_file'], subset['time_taken'], marker='o', label=f'{attack_type.capitalize()} Attack')

plt.xlabel('Hash File')
plt.ylabel('Time Taken (seconds)')
plt.title('Time Taken by Attack Type')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
