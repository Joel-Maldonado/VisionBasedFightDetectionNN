
  for img in vid:
    print(img)
    img = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
    cv2.imshow('video', img)
    cv2.waitKey(0)