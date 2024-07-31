### Нужно поменять config.yaml для работы preprocessing
# Структура проекта:
 - StudCampAE-256/128 - ноутбук для тренировки автоэнкодера.
 - Evaluation - scores, feature selection, pca analysis
 - Classic models - scores, features selection, models
 - StatTests - делает xlsx файл со сравнением двух групп, для этого нужно только csv, надо поменять kaggle/input на свои данные.
 - Dataloader - загрузка данных в torch тензоры
 - Generate data 3d volumes - генерация аугментированных данных

# Как работали с данными?
- Балансировка классов:
  - SMOTE
  - Аугментация (SVD, отражения, сдвиги)
 
# Какие модели?
 - Catboost
 - GradientBoosting
 - RandomForest
 - LogisticRegression

# Как отобраны признаки(110 -> ~20)?
- Статистические тесты на предмет разницу между группами (pvalue < 0.05 -> группы сильно отличаются)
- SelectKBest(scikit-learn)
- Сравнение по feature importance со случайно сгенерированными признаками

# Почему AutoEncoder?
 - 4M/16M параметров модель, обучающаяся восстанавливать изображения из их сжатого "латентного" представления, не требует разметки классов в классическом виде, сложно переобучить.
 - Используются не полные МРТ сканы мозга, а лишь малая часть этих сканов, содержащая опухоль и симметрично дополненные нулями до фиксированного размера.
 - Можно использовать все данные для обучения, а не только размеченные.
 - Обучается с дополнительным triplet loss с косинусной мерой близости, благодаря чему обучается *раздивигать границу классов*, делая предсказания на этих признаках *более доверительными*
   - Малый размер модели и достаточный размер train разметки очень важен, чтобы она не переобучилась на этот loss и генерализировалась на out-of-train данные.
   - Модель учится разделять классы на два перпендикулярных направления, что можно увидеть ниже.
   - Пример, как модель справилась с неразмеченными данными, в сравнение с классическими radiomics признаками на тех же данных:
<p float="right">
  <img src="/pcalatents.png" width="300" />
  <img src="/pcastandard_done_right.png" width="300" /> 
</p>

   - Классы могут пересекаться, так как в обоих есть люди с мутацией/без мутации, группа >700 состоит из опухолей преимущественно без мутаций. 
   
### Таким образом, имея больше размеченных данных можно обучить классификатор на латентных представлениях энкодера или дополнить классические признаки.
