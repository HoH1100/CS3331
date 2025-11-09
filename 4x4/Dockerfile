# Sử dụng Python base image
FROM python:3.10-slim

# Tắt buffer để log realtime
ENV PYTHONUNBUFFERED=1

# Cài đặt thư viện hệ thống cần thiết
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy file yêu cầu thư viện
COPY requirements.txt .

# Cài package Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy code vào container
COPY . .

# Chạy file Python của bạn mặc định (sửa lại cho phù hợp)
CMD ["python", "tic_tac_toe_rl.py"]
