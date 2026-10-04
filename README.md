*INPUT*

a image: x

a CNN model: f(x)

and a objective class: c

*OUTPUT*

Given an input image \(x\), a CNN \(f(x)\), and a target class \(c\), Grad-CAM produces a class-discriminative localization map \(L_{\text{Grad-CAM}}^c\) highlighting the spatial regions that contribute to the model's score for class \(c\).a map that indicates which regions where the most important to get the c score. 

Si modificáramos ligeramente cada una de esas activaciones, ¿cómo cambiaría el score de "imagen"?
