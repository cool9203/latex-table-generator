# Latex table generator

## Pre-install

```bash
sudo apt install wkhtmltopdf fonts-noto-cjk
cd scripts
bash install-pandoc.sh
cd ..
```

## How to use

Reference: [run.sh](/scripts/run.sh)

## Note

### Dockerfile

Dockerfile 沒有實際 build 起來看是否可以正確使用

### 其他渲染 html 的 package

- [imgkit](https://github.com/jarrekk/imgkit) 目前使用的
- [WeasyPrint](https://github.com/Kozea/WeasyPrint) 沒有測試過，看起來是最可行的另一個
- [wand](https://github.com/emcconville/wand) 基於 [ImageMagick](https://github.com/ImageMagick/ImageMagick)，看起來是最可行的另一個
- [html2image](https://github.com/vgalin/html2image) 但該 package 缺點是沒辦法擷取全頁大小，這樣用起來很不好用
- [WeasyPrint](https://github.com/Kozea/WeasyPrint) 沒有測試過
- [pyppeteer](https://github.com/pyppeteer/pyppeteer) 沒有測試過，但問題應該與 html2image 相同
- [CairoSVG](https://github.com/Kozea/CairoSVG) 看起來只能轉 SVG，但不確定
- [dom-to-image](https://github.com/tsayen/dom-to-image) 需要配合 browser
