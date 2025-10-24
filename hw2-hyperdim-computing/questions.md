### Part A: Hyperdimensional Computing [8 pts total, 2 pts / question]

**Task 1:** In `hdc.py`, implement the hypervector generation, binding, bundling, Hamming distance, and permutation operations in the scaffold code. Also, implement the `add` and `get` functions in the item memory `HDItemMem` class; these functions will be used to manage the atomic hypervectors. Based on them, implement an HD encoding procedure that encodes strings in `make_word` function. You should also implement the `make_letter_hvs` function. It should produce a codebook for the letters, which serves as the first argument for the `make_word` function. Refer to the main scripts for how these two functions are invoked in sequence.

For example "fox" would be translated to sequence ["f","o", "x"]. For simplicity, use a hypervector size of 10,000 to answer these questions unless otherwise stated.

**Q1.** Construct a HDC-based string encoding for the word "fox". How did you encode the "fox" string? How similar is the hypervector for "fox" to the hypervector for "box" in your encoding? How similar is the hypervector for "xfo"? How similar is the hypervector for "car"? Please remark on the relative similarities, not the absolute distances.
````
---- terminal output ----
0.2514
0.4943
0.4916
````
I encode fox string as 'f' [op-bundle] ([op-permute] 1 'o') [op-bundle] ([op-permute] 2 'x')
The words "fox" and "box" are more similar than "fox" with "xfo" and with "car." The former pair has a hamming distance of 0.25, whereas the latter two pairs 0.5.

**Q2.** Change your encoding so the order of the letters doesn't matter (apply the changes for this question only). What changes did you make? Please remark on the relative similarities, not the absolute distances.
```
---- terminal output ----
0.2446
0.0
0.4928
```
I eliminated the use of permutation in each character, so that the result now is a set of bundled characters and order is discarded. "fox" now has a distance of 0 to "xfo", suggesting that they are essentially the same hypervector. The distance between "fox" and "box" and between "fox" and "car" remains relatively the same. 

-------

**Task 2:**: Implement the bit flip error helper function (`apply_bit_flips`). Then apply bit flip errors to hypervectors before the distance calculations, where the bit flip probability is 0.10. Specifically, before computing the distance between two hypervectors hv1 and hv2, you may apply the bit flip error to one of them, say hv1. Use the `monte_carlo`, `study_distributions`, and `plot_hist_distributions` helper functions to study the distribution of distances between `fox` and `box`, compared to the distance between `fox` and `car` with and without hardware error.

**Q3.** Try modifying the hardware error rate (`perr`) with fixed hypervector size `10000`. How high can you make the hardware error until the two distributions begin to become visibly indistinguishable? What does it mean conceptually when the two distance distributions have a lot of overlap?

The hardware error can become as high as 0.5 when the two distribution begin to become visibly indistiguishable. When the two distance distributions have a lot of overlaps, it means that the two words are equally similar / different from "fox".

**Q4.** Try modifying the hypervector size (`SIZE`) with fixed hardware error rate `0.10`. How small can you make the word hypervectors before the two distributions begin to become visibly indistinguishable?

The word hypervector size can becomes as small as 6 or 7 before the distributions begin to become visible indistinguishable.

-----

**Task 3:**: Next, fill out the stubs in the item memory class -- there are stubs for threshold-based (`matches`) and winner-take-all (`wta`) queries, and for computing the Hamming distances between item memory rows and the query hypervector. The item memory class will be used in later exercises to build a database data structure and an ML model.

### Part B: Item Memories [`hdc-db.py`, 10 points total, 2 pts / question]

Next, we will use this item memory to implement a database data structure; we will be performing queries against an HDC-based database populated with the digimon dataset (`digimon.csv`). The HDDatabase class implements the hyperdimensional computing-based version of this data structure, and contains stubs of convenience functions for encoding / decoding strings and database rows, as well as stubs for populating and querying the database. We will implement this class and then invoke `build_database` to build a database using the HDDatabase class. For simplicity, use a hypervector size of 10,000 to answer these questions unless otherwise stated.

_Tip_: For this exercise, map every distinct string to an atomic random hypervector. This will keep retrieval tasks relatively simple. For decoding operations, you will likely need to use the self-inverse property of binding and perform additional lookups to recover information.

---------

__Task 0__: The database data structure contains multiple rows, where each row is uniquely identified with a primary key and contains a collection of fields that are assigned to values. In the HD implmentation, the database rows are maintained in item memories, and row data is encoded as a hypervector. Decide how you want to map the database information to item memories. Implement the database row addition `add_row` function, which should invoke the `encode_row` helper function and update the appropriate item memories.

---------

__Task 1__: Implement the string and row encoding functions (`encode_string`, `encode_row`). These encoding functions accept a string and a database row (field-value map) respectively, and translate these inputs into hypervectors by applying HD operators to atomic basis vectors. Then, implement the string and row decoding functions (`decode_string`, `decode_row`) which take the hypervector representations of a string or database row respectively and reconstructs the original data. The decoding routines will likely need to perform multiple codebook / item memory lookups and use HD operator properties (e.g., unbinding) to recover the input data. Execute `digimon_test_encoding` function to test your string and row encoding routines and verify that you're able to recover information from the hypervector embedding with acceptable reliability.

**Q1.** Describe how you encoded database rows as hypervectors. Write out the HD expression you used to encode each piece of information, and describe any atomic hypervectors you introduced.

I encoded each distinct string (both field key and field value) as atomic hypervectors. 
Then, with each row contains information in the format of 
{ k1 : v1 , k2 : v2 , .... kn : vn}
I encoded it as
(k1 [op-bind] v1) [op-bundle] (k2 [op-bind] v2) [op-bundle] .... [op-bundle] (kn [op-bind] vn)

**Q2.** Describe how you decoded the strings / database rows from hypervectors. Describe any HD operations you used to isolate the desired piece of information, and describe what item memory lookups you performed to recover information. If you're taking advantage of any HD operator properties to isolate information, describe how they do so.

For decoding string with a hypervector, I just uses winner-take-all query functions I implemented previously.
For decoding database row, I first need auxiliary information on the fieds keys. Therefore, I stores a copy of the keys when first inserting rows into the database. Provided a encoded row hypervector, I iteratively unbind the hypervector with each field key, and find the string closest to the unbind hypervector.

--------

__Task 2__: Next, we'll implement routines for querying the data structure. Implement the `get_value` and `get_matches` stubs -- the `get_value` query retrieves the value assigned to a user-provided field within a record. The `get_matches` stub retrieves the rows that contain a subset of field-value pairs. Implement both these querying routines and then execute `digimon_basic_queries` and `digimon_value_queries` to test your implementations.

**Q3.** How did you implement the `get_value` query? Describe any HD operators and lookups you performed to implement this query.

I first obtain the encoded row information based on the key specified.
Then, to find the specified field, I first encode field to its corresponding hypervector. I perform unbind operation between the encoded row and the encoded field. This should give a hypervector that is close to the value hypervector being previously binded to the field hypervector. I then decode the value hypervector by using winner-take-all approach to find the closest string to it.

**Q4.** How did you implement the `get_matches` query? Describe any HD operators and lookups you performed to implement this query. Try using lower threshold values. How high of a distance threshold can you set before you start seeing false positives in the returned results?

For get matches, I first encode the field value dict into a row hypervector. I then leverage mathces function, which iterates through all row keys and return any key whose row hypervector has a hamming distance to the query hypervector that is smaller than the pre-specified threshold.
For the first virus-plant query, I started seein false positives in a threshold of 0.43. For the second champion query, the false positives appear around a threshold of 0.50.

-----

__Task 3__: Implement the `get_analogy` query, which given two records and a value in one of the records, identifies the value that shares the same field in the other record as the input value. For example, if you perform an `analogy` query on the `US` and `Mexico` records, and ask for the value in the `Mexico` record that relates to the `Dollar` value in the `US` record, this query would return `Peso`. This query completes the analogy  _Dollar is to USA as <result> is to Mexico_. Execute `analogy_query` to test your implementation.

_Tip_: If you want more information on this type of query, you can look up "What We Mean When We Say 'What's the Dollar of Mexico?'"

**Q5.** How did you implement the `get_analogy` query? Describe how this is implemented using HD operators and item memory lookups. Why does your implementation work? You may want to walk through and HD operator properties you leveraged to complete this query.

I first find the encoded row hypervector for both the target and the other key. In the current encoding scheme, I bind each value and key tuple. Based on the commutative and self-inversable property of bind operation, the key and vlaue are symmetric in this encoding representation. Therefore I use the same method as `get_value` function to obtain the field key corresponding to the target value.
Specifically, with the target row hypervector, I first unbinds it with the target value, then uses winner-take-all approach to find the string closest to the resulting hypervector.
After finding the field key of interest, I essentially repeat `get_value` function procedure to retrieve the value associated with this field key in the other-key row.

### Part C: Implementing an HDC Classifier [10 pts total, 2 pts/question, hdc-ml.py]

Next, we will use an item memory to implement an MNIST image classifier. A naive implementation of this classifier should easily be able to get ~75% accuracy. In literature, HDC-based MNIST classifiers have been shown to achieve ~98% classification accuracy with model size of only a few bytes while being error resilient. In this exercise, you will implement the necessary encoding/decoding routines, and you will implement both the training and inference algorithms for the classifier.

__Tips__: Try a simple pixel/image encoding first. For decoding operations, you will likely need to use the self-inverse property of binding to recover information.

-------------

**Task 1**: Fill in the `encode_pixel`, `decode_pixel`, `encode_image`, and `decode_image` stubs in the MNIST classifier. These functions should translate pixels/images to/from their hypervector representation. Then use the `test_encoding` function to evaluate the quality of your encoding. This function will save the original image to `sample0.png`, and the encoded then decoded image to `sample0_rec.png`.

**Q1.** How did you encode pixels as a hypervector? Write out the HD expressions, and describe what atomic/basis hypervectors you used for the encodings. 

There are three pieces of information in a pixel: row number, column number, and pixel value.
In the current image representation, all pixels are binary, which means that we only need to take into account representing '0' and '1' bit.
I create two atomic hypervectors, one for 1 encoding and one for 0 encoding. Based on the pixel value at each coordinate I use the corresponding atomic hypervector.
There is a way to serialize row and column information by representing the location as loc = row * col-size + col. However, this requires dependency knowledge on the size of the image. To avoid this dependency, row and column information are encoded separately in this scheme. A row atomic hypervector and a column atomic hypervector are created for this representation. Based on the row number and column number, the row and column hypervectors are permuted by the offset.
Finally, to aggregate all information in a readily retrievable form, I first bind row and column information together into a location hypervector. I then bind the pixel value hypervector to the location hypervector. The resuling encoding is as follows:

Encode(pixel, row, col) = atomic-1 / atomic-0 [op-bind] (([op-permute] row-number atomic-row) [op-bind] ([op-permute col-number atomic-col]))

**Q2.** How did you encode images as a hypervector? Write out the HD expressions, and describe any atomic/basis hypervectors in the expression. 

To ensure that the final encoding is relatively close to each of the pixel-wise representation, I bundle individual pixel-level information together, so that the HD expression is

Encode(image) = Encode(pixel[0,0], 0, 0) [op-bundle] Encode(pixel[0,1], 0, 1) [op-bundle] Encode(pixel[0,2], 0, 2) ... [op-bundle] Encode(pixel[1,0], 1, 0) .... [op-bundle] Encode(pixel[n-1,n-1], n-1, n-1) 

No additional atomic/basis hypervector is introduced in this step.

-----------------------

**Task 2**: Fill in the `train` and the `classify` stubs in the MNIST classifier. Test your classifier out by invoking the `test_classifier` function. What classification accuracy did you attain? You should be able to achieve ~75% with the standard techniques learned in class. We will give full credit for accuracy about or above 75%.

**Q3.** What happens to the classification accuracy when you reduce the hypervector size; how small of a size can you select before you see > 5% loss in accuracy? 

The classification accuracy first increases to around 78%, but then gradually decreases. When the size is around 2000-3000, I start with see > 5% loss in accuracy.

**Q4.** What happens to the classification accuracy when you introduce bit flips into the item memory's distance calculation? Note that you shold only apply bit flips during inference. How much error can you introduce before you see a > 5% loss in accuracy?

The accuracy gradually decreases as bit flip erros are introduced. Around 0.2 bit errors would result in a >5% accuracy loss.

------------------------

**Task 3**: You can also implement a simple generative model using hyperdimensional computing. In this following exercise, we will use HDC to generate images of handwritten digits from the classifier label. Naive HD generative models are very similar to HD classifiers, and are constructed in two simple steps:

- _Constructing a Generative Model._ For each classifier label, group training data by label, and translate each datum to a hypervector. Next, generate a probability vector for each label. The probability value at position i of the probability vector is the probability that the hypervector bit in position i is a "1" bit value. The probability vector can easily be computed by summing up the M hypervectors that share the same label, and then normalizing by 1/M.

- _Generating Random Images._ To generate a random image for some label, you sample a binary hypervector from the probability hypervector for that label. You then translate the hypervector to an image using the hypervector decoding routine (`decode_image`). You can sample and bundle multiple hypervectors to average the result.

You can find a sample of generated image `sample_generated.png` for number `7` in the folder.

**Q5.** Use `test_generative_model` function to test your generative model. Include a few pictures outputted by your generative model in your submission.

I stored a few generated images inside the "generated-images" folder in the submission.
