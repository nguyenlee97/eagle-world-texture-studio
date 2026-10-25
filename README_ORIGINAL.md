# Eagle World Texture & Landmark Studio (`tools/world-texture-studio`)

Công cụ trực quan quản lý, xem trước 2D/3D, chuyển đổi (convert) theo từng hạng mục bề mặt công trình và đúc tượng đài 3D tĩnh cho **`C:\Users\LENOVO\Desktop\GTA\Project Eagle\gta sa gameplay-lab`**.

## Cách khởi chạy

Chạy file:
- `Start_World_Texture_Studio.bat` (mở trình duyệt tại `http://127.0.0.1:8950`)
- Hoặc chạy trực tiếp bằng lệnh:
  ```powershell
  python tools\world-texture-studio\app.py
  ```

## Kiến trúc 3 Phân hệ chính

1. **Mod Texture Bề Mặt 2D (`CAT1..CAT4` — Chỉ sửa `.TXD`, giữ nguyên `.DFF`, có xem trước trên mô hình 3D `.DFF` thực tế)**:
   - **Converter A (`CAT1_BILLBOARDS`)**: Đọc tỷ lệ khung 3D thực tế từ `.DFF` để tự động bù giãn UV (chống méo hình khi raster lưu vuông `1:1` nhưng khung biển ngoài đời là `2:1`), nén `DXT1` + Mipmaps (`512×256` / `512×512` / `1024×512`) và **tự động thu nhỏ đồng bộ luôn ảnh `LOD` (`64×32` / `64×64` `DXT1`)** vào file `*_lod.txd` cặp đôi.
   - **Converter B (`CAT2_MURALS_BANNERS_GRAFFITI`)**: Hòa trộn vân gạch/bê tông từ bức tường gốc (`Wall-Grain Blend`) để tranh anime nhìn như sơn trên tường thật; hỗ trợ `DXT5` Alpha cho Graffiti (`tags_la*.txd`) và hiệu ứng nếp vải cho Banner dọc (`cj_banner.txd`).
   - **Converter C (`CAT3_NEON_LED_SCREENS_CASINOS`)**: Thay toàn bộ màn hình hoặc **chỉ thay đúng 1 ô vùng con (`Sub-Region / UV Island`)** trên tấm Atlas ghép nhiều bảng hiệu; tùy chọn phủ lưới bóng đèn LED Scanline & tăng độ rực ban đêm.
   - **Converter D (`CAT4_POSTERS_SHOPS_TRANSIT_VENDING`)**: Tự động khử độ kéo giãn UV của khung poster nhà chờ xe buýt (`bustopm.txd:CJ_BS_MENU5`), giữ lại viền khung kim loại/nút bấm máy bán nước (`vendmach*.txd`), và phủ bóng kính phản chiếu.

2. **Xưởng Đúc & Thay Tượng Đài 3D (`CAT5_MONUMENTS` — Thay cả `.DFF` + `.TXD` tĩnh)**:
   - Duyệt toàn bộ `93` mô hình Tượng đài & Biểu tượng 3D (`missliberty`, `sphinx01`, `atrium`, `big_cock`, `pirateship01`, `disney`...).
   - Cho phép chọn bất kỳ nhân vật anime nào từ kho **`Skin Library`** (`1,638` skins) hoặc tải lên cặp `.DFF` + `.TXD` mới.
   - Tự động gỡ bỏ `Skin PLG` / `HAnim PLG` xương động, xoay trục chuẩn Z-up, căn chỉnh chiều cao & tâm chân đế khớp với khung bao (`Bounding Box`) của tượng gốc, tạo **Bệ Đá Bát Giác 2 Tầng**, phủ chất liệu tượng (`Đá Cẩm Thạch Trắng`, `Vàng 24K`, `Đồng Cổ`, `Ngọc Bích`, hoặc giữ nguyên màu `Anime Figure`), và nướng màu đỉnh Ngày/Đêm (`PRELIT`).

3. **Staged Mod Pack & Safe `.IMG` Deploy / Bit-for-Bit Rollback**:
   - Mọi chỉnh sửa được lưu trước vào `data/staging/` để xem trước cả 2D lẫn 3D WebGL.
   - Khi bấm **Deploy**, công cụ kiểm tra game đã tắt, giữ khóa ghi độc quyền `C:\Users\LENOVO\Desktop\GTA\data\project-eagle\coordination\game-write.lock`, đồng bộ vào tất cả các kho `.IMG` đang mount chứa member đó (`GROUP_B1_TEXTURES.IMG`, `GROUP_A_MODS.IMG` và kho gốc `S&S`), và trích lưu đúng vùng sector gốc để **Rollback Bit-for-Bit** trong 1 cú nhấp chuột.
