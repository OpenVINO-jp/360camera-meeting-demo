■ 概要

360度カメラを利用した遠隔地とのテレビ会議の利用シーンにおける、OpenVINOを使ったデモプログラムです。
以下のユースケースを想定して作成しています。

・画面の分割機能
・話者推定
・ホワイトボードOCR

利用する動画によって使用するモデル、閾値（pd_confidence,fd_confidence）の調整が必要です。
これはカメラの位置、高さ、参加人数、会議室の大きさ、机の高さなどで適切なパラメータが変化するためです。

■ ホワイトボードOCR

ホワイトボードOCRについては、Google Vision APIとOpenVINOの双方が利用可能です。
実行にあたり、CloudVisionAPIを事前に使える状態にしておいてください。
OpenVINOでの解析は動画の状況、キャプチャのタイミングで期待通りの結果が得られない可能性があります。
ズームを行わない、キャプチャを何回か試してみる、２値化の閾値を変更してみる、文字を変更してみるなど、細かな調整が必要となります。

■ 借用ソース

以下のソースを利用しています。ありがとうございます。
https://github.com/yas-sim/openvino-wrapper
https://github.com/yas-sim/openvino_open_model_zoo_toolkit

■ 使用モデル

intel/face-detection-adas-0001
intel/face-detection-retail-0005
intel/facial-landmarks-35-adas-0002
intel/handwritten-japanese-recognition-0001
intel/pedestrian-detection-adas-0002
intel/person-detection-retail-0013
intel/person-reidentification-retail-0288
intel/text-detection-0003

■ 実行環境

以下の環境で確認をおこないました。
Windows 10 Home / OpenVINO 2021 / Python 3.7.9

■ 実行

pyrhon meeting_demo.py　で実行します。（windowsの場合）

■ キーボード操作

実行時に以下のキーボード操作を受け付けます

t   ... toggle window mode.
        starting -> meeting -> 2x2 -> starting ..

1-4 ... zoom person at meeting mode.
        "1" key is pushed, zoom to no.1 person

w   ... zoom white-board
i   ... reading white-board by OpenVINO
g   ... reading white-board by CloudVisionAPI

p   ... show person detect
f   ... show face detect
r   ... show detected(croped) rect

l   ... show face-landmark

■ re-identificationについて

ミーティングに参加している人のデータについては、meeting_member.ymlに定義しています。
meeting_memberに対応した顔データとして、\rsc\face配下にデータを格納しています。フォルダ配下のファイル名は問いません。

画像を採取するオプションは付けていませんが、以下の方法で増やす事ができます。
1. meeting_demo.py - def main()内の、f_reid.PreloadImage() ## preload files for build db　をコメントアウトする。
2. reidentification.py - def fncReid()内の、  #cv2.imwrite(face_dir + str(feature_db[-1]['time']) + '.jpg', feature_db[-1]['img']) のコメントを外す
3. meeting_demo.py実行すると、rsc/face配下に顔データが保存されます。手作業でフォルダに配置してください。
