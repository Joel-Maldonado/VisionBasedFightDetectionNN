
    #   y1, x1, y2, x2 = box_model.predict(tf.expand_dims(gray, 0))[0]

    #   print(y1, x1, y2, x2)

    #   y1 = int(y1 * IMG_SIZE)
    #   x1 = int(x1 * IMG_SIZE)
    #   y2 = int(y2 * IMG_SIZE)
    #   x2 = int(x2 * IMG_SIZE)

    #   cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    