# Lab A: DCT watermark baseline trong MPEG video

## Mục tiêu
Bạn sẽ tạo MPEG video, nhúng watermark bản quyền bằng DCT mid-frequency, detect watermark và đo PSNR/BER.

## Lệnh bắt buộc

```bash
cd /home/ubuntu
./generate_source.sh
python3 wm_tool.py embed --config wm_config.json --input videos/source.mpg --output videos/watermarked.mpg --schedule out/key_schedule.json
python3 wm_tool.py extract --config wm_config.json --input videos/watermarked.mpg --schedule out/key_schedule.json
./metrics.sh
checkwork
