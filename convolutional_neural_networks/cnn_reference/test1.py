import cv2


img = cv2.imread('/Users/*****/Downloads/temp/gopher1.jpg', 0)
rows,cols = img.shape

print("rows: %g" % rows)

#cv2.imshow('image',img)