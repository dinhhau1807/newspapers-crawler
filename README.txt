Nhóm: 04
Thành viên:	1760006 - Nguyễn Trần Tuấn Anh
	        1760056 - Nguyễn Đình Hậu
	        1760075 - Đặng Quốc Huy
====================================================
** CÁCH SỬ DỤNG

+ Crawl dữ liệu
- Tại folder crawler
- Chạy chương trình: scrapy crawl spiderman
-> Dữ liệu crawl được sẽ được lưu vào folder NEWSPAPERS

+ Xử lý dữ liệu để tính toán độ chính xác
- Tại folder process
- Sau khi crawl dữ liệu, copy các folder chủ đề đã crawl được cần tính (bên folder NEWSPAPERS của crawler) vào folder input.
- Chạy chương trình: py main.py 
-> Dữ liệu được xử lý sẽ được lưu vào folder output