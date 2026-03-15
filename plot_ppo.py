import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_tianshou_metrics(csv_path, output_dir='reports/tianshou_ppo_plot_10_2_random'):
    # 1. Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục: {output_dir}")

    # 2. Đọc dữ liệu
    try:
        df = pd.read_csv(csv_path)
        print(f"Đã đọc file {csv_path} với {len(df)} dòng dữ liệu.")
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    # Xác định trục X (Episode hoặc Step)
    if 'episode' in df.columns:
        x_axis = 'episode'
    elif 'step' in df.columns:
        x_axis = 'step'
    else:
        df['index'] = df.index
        x_axis = 'index'

    # Danh sách các chỉ số cần vẽ (dạng số)
    numerical_metrics = ['ep_reward', 'avg_speed', 'total_energy', 'wiggle', 'safety', 'success']
    
    # 3. Vẽ các chỉ số dạng số
    for metric in numerical_metrics:
        if metric in df.columns:
            plt.figure(figsize=(12, 6))
            # Vẽ đường gốc với độ mờ thấp (alpha=0.3)
            sns.lineplot(data=df, x=x_axis, y=metric, alpha=0.3, label=f'{metric} (raw)')
            
            # Vẽ đường trung bình trượt (rolling mean) để dễ nhìn xu hướng
            # Cửa sổ trượt (window) tự điều chỉnh theo độ dài dữ liệu
            window_size = max(1, min(50, len(df)//10))
            df[f'{metric}_smooth'] = df[metric].rolling(window=window_size, min_periods=1).mean()
            sns.lineplot(data=df, x=x_axis, y=f'{metric}_smooth', linewidth=2, label=f'{metric} (smooth)')
            
            plt.title(f'Biểu đồ {metric} theo thời gian', fontsize=15)
            plt.xlabel('Thời gian (Episode/Step)', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            # Lưu ảnh
            file_name = f"{metric}_plot.png"
            plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
            plt.close()
            print(f"Đã lưu đồ thị: {file_name}")

    # 4. Xử lý riêng cho cột 'reason' - ĐÃ CẬP NHẬT CHIA 30 CỘT
    if 'reason' in df.columns:
        num_bins = 30 # Số lượng cột mong muốn
        plt.figure(figsize=(15, 7)) # Tăng chiều rộng để hiển thị 30 cột rõ hơn
        
        # Chia dữ liệu thành 30 bins (khoảng)
        # Sử dụng labels ngắn gọn (P1, P2...) để tránh rối mắt
        df['bin'] = pd.cut(df.index, bins=num_bins, labels=[f"P{i+1}" for i in range(num_bins)])
        reason_counts = df.groupby(['bin', 'reason']).size().unstack(fill_value=0)
        
        # Vẽ biểu đồ cột chồng (stacked bar)
        reason_counts.plot(kind='bar', stacked=True, figsize=(15, 7), colormap='viridis')
        
        plt.title(f'Phân bổ lý do kết thúc (Reason) qua {num_bins} giai đoạn', fontsize=15)
        plt.xlabel('Giai đoạn tập luyện (Part 1 - 30)', fontsize=12)
        plt.ylabel('Số lượng (Count)', fontsize=12)
        plt.xticks(rotation=45) # Xoay nhãn 45 độ để dễ đọc
        plt.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        
        file_name = "reason_distribution_plot.png"
        plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
        plt.close()
        print(f"Đã lưu đồ thị: {file_name}")

# Đường dẫn file CSV của bạn
csv_file = 'reports/tianshou_td3/training_log_14032026_224702.csv'
plot_tianshou_metrics(csv_file)