# Latex table generator

## Pre-install

```bash
sudo apt install wkhtmltopdf
cd scripts
bash install-pandoc.sh
cd ..
```

## Note

### Dockerfile

Dockerfile 沒有實際 build 起來看是否可以正確使用

### 其他渲染 html 的 package

[imgkit](https://github.com/jarrekk/imgkit) 目前使用的
[html2image](https://github.com/vgalin/html2image) 但該 package 缺點是沒辦法擷取全頁大小，這樣用起來很不好用
