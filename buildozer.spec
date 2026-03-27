[app]
title = BTC 5分钟预测
package.name = btc5min
package.domain = org.btc
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 1.0

requirements = python3,kivy,requests,pandas,numpy,scikit-learn

# 禁用不需要的依赖
exclude_patterns = models/*

# 权限配置
android.permissions = INTERNET

# 屏幕方向
android.orientation = portrait

# 图标和启动屏幕
# android.icon = %(source.dir)s/icon.png
# android.splash = %(source.dir)s/splash.png

# 构建配置
android.api = 33
android.minapi = 21
android.sdk = 24
android.ndk = 25b
android.arch = armeabi-v7a,arm64-v8a

# 其他配置
fullscreen = 0

[buildozer]
log_level = 2
warn_on_root = 1