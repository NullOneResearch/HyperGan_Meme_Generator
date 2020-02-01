# HyperGan_Meme_Generator
Генератор мемов на основе HyperGan.

Проект получился не столько кодинговый, сколько сторителлинговый.
Использовалась уже готовая нейросеть HyperGan, скачать её можно здесь - https://github.com/HyperGAN/HyperGAN

Также в тетрадке Jupyter Notebook находятся полезные команды, которые помогут при установке CUDA, TensorFlow и самой HyperGan.

Цель - Создать машинные абстрактные мемы.

Задачи:
1. Создание выборки мемных сообществ 
2. Парсинг сообществ 
3. Ресайз изображений 
4. Прогон  изображений через нейросетку 
5. Оценка результатов обучения.

Заранее стоит отметить, что HyperGan - это векторная нейросеть, так что она плохо подходит для решений задач с текстом, но неплохо - для решения задач визуала.

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/Пикча1.png)

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/Пикча2.png)

В рамках проекта использовалась видеокарта NVIDIA GeForce GTX 1060 Max-Q.
Для работы нейросети требовалась установка CUDA, чтобы можно было вести обучение на видеокарте. У разных видеокарт поддержка разных версий CUDA, также для нейросети требовалась определенная версия Tensorflow. При этом, она также должна была сочетаться с версией CUDA, которая должна поддерживаться видеокартой, и всё это должно было сочетаться с HyperGan. 

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/Пикча3.png)

Обучение выглядит вот так, и идет в реальном времени. Можно визуально оценить состояние нейросети, либо через функцию потерь.

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0008.jpg)

В определеннный момент это начало выглядеть жутковато, особенно если учитывать, что работа шла глубокой ночью.

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0009.jpg)

Также на картинках стали появляться глаза. 
Это коррелирует с исследованиями о том, что человек ассоциирует себя с чем-то именно в первую очередь по наличию глаз, и этот паттерн нейросеть определила в мемах и воспроизвела.

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0010.jpg)

Если проявить немного воображения, то на картинках можно увидеть то, что похоже на мемы. Но это была еще ранняя стадия обучения нейросети.

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0011.jpg)
![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0012.jpg)

После получения таких результатов с мемов самых популярных пабликов было понятно, что требуется более однородная выборка. Так что в последствии была сделана выборка из Trollface-пабликов, отсуда были спарсены картинки.

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0013.jpg)
![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0014.jpg)
![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0015.jpg)

Также при ресайзе терялась часть текста. К сожалению, на обучение в более высоком качестве мощностей видеокарты не хватило.

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0016.jpg)

В качестве альтернативы HyperGan рассматривалась PixelCnn. Различия состоят в том, что первая нейросеть является векторной, и хорошо воспроизводит именно формы объектов, а вторая заточена на распознавание именно паттернов пикселей, так что PixelCnn должна лучше работать с текстом и воспроизводить его.
Ознакомиться с ней можно по ссылке - https://github.com/openai/pixel-cnn

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0017.jpg)

Тем не менее, впоследствии выяснилось, что для её нормальной работы требуется 8 видеокарт, так что от идеи отказались. На одной видеокарте она не заработала даже в минимальном режиме.

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0018.jpg)
![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0019.jpg)

В дальнейшем выдача нейросети стала больше похожа на мемы. Тем не менее, они оставались жутковатыми.

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0021.jpg)
![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0022.jpg)
![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0023.jpg)
![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0024.jpg)
![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0025.jpg)
![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/26.jpg)

В качестве итога можно сказать, что для создания нормальных мемов требуются большие мощности видеокарт. Проект будет повторён после того, как они будут получены.

И, да:

![Image alt](https://github.com/NullOneResearch/HyperGan_Meme_Generator/raw/master/images/0029.jpg)
