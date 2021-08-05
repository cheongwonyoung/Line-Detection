import numpy as np
import cv2

cap = cv2.VideoCapture("walk.avi") # 비디오 캡쳐 객체 생성
if cap.isOpened() == False: raise Exception("영상 연결 안됨") # 예외처리

# 일시정지 기능 (마우스 좌클릭)
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  
        cv2.waitKey(0)

mask = np.zeros((360,640)) # 이미지와 같은 사이즈(흑)
imshape = mask.shape  # 마스크 사이즈 저장(행,열)

# 레일영역을 잡을 좌표값들
vertices = np.array([[(0,imshape[0]-120),
                     (0,imshape[0]),
                    (imshape[1],imshape[0]),
                    (imshape[1],imshape[0]-120),
                    (imshape[1]/2,10)]], dtype=np.int32)

# 좌표값들로 마스크에 그리기
cv2.fillPoly(mask, vertices, 255) # mask에 vertices 배열 좌표로 다각형 도형 그리기
mask=mask.astype(np.uint8) # 타입 변경

cv2.namedWindow("Source")  # 원본 이미지 윈도우 생성
cv2.namedWindow("just gray")  # 그레이스케일 변환된 이미지 윈도우 생성
cv2.namedWindow("blur_gray")  # 블러처리된 이미지 윈도우 생성
cv2.namedWindow("edges")  # 케니엣지 적용된 이미지 윈도우 생성
cv2.namedWindow("mask")  # 원하는 영역 마스크 이미지 윈도우 생성
cv2.namedWindow("masked")  # 마스크 영역 내의 엣지 이미지 윈도우 생성
cv2.namedWindow("line")  # 검은 배경에 검출된 라인 그린 이미지 윈도우 생성
cv2.namedWindow("result")  # 결과 이미지 윈도우 생성

while (True): # 계속 재생 ( 영상 끝날때 까지)
    ret, src = cap.read() # 비디오를 한 프레임씩 읽음.
                            # 제대로 읽으면 ret에 True, 실패하면 False.
                            # src에는 읽은 프레임이 들어감
    if not ret: break # 읽지 못했을 상황 예외처리
    src = cv2.resize(src, (640, 360)) # 크기 조절

    # 중심값
    x_center = src.shape[0] // 2
    y_center = src.shape[1] // 2

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) # grayscale
    blur_gray = cv2.GaussianBlur(gray, (5,5), 0) # blur 처리 //(5,5)는 커널사이즈
    canny_edges = cv2.Canny(blur_gray, 50, 200) # canny 적용

    masked_image = cv2.bitwise_and(canny_edges, mask) # 비트 연산을 통해 원하는 영역만 지정

    hough_lines = cv2.HoughLinesP(masked_image, 2, np.pi / 180, 95, np.array([]), 120, 150)

    line_img = np.zeros_like(src, dtype=np.uint8) # 검출된 라인을 그릴 검은 배경

    grad = 0.0  # 직선들의 기울기 합
    line_count = 0 # 한번 그릴때 그려지는 직선의 수 총합

    for line in hough_lines:
        for x1,y1,x2,y2 in line: # 직선의 시작점과 끝점의 좌표 값
            if x2-x1 !=0: # 분모가 0이 되는 예외상황 처리
                temp = (y1-y2)/(x2-x1) # 직선의 기울기를 구하는 코드 ( y1과 y2는 값이 커질수로 음의 방향임으로 반대로 써줌)

            cv2.line(line_img, (x1,y1), (x2, y2), 255, 10) # 직선 그리기

            line_count += 1 # 그려진 직선의 개수 체크
            grad += temp # 직선의 기울기들의 총 합을 구하기

            if line_count == 5: # 5개 직선이 그려졌을때 방향 알려주기 (적당한 출력을 위해 평균적으로 그리는 직선 수 만큼 계산 후 출력.)

                if grad < -2 : # 직선들의 기울기의 합이 -2보다 작을 때 왼쪽으로 가라고 알려주기
                    print("left")
                elif grad > 2: # 직선들의 기울기의 합이 2보다 클 때 오른쪽으로 가라고 알려주기
                    print("right")
                else: # 기울기가 -2와 2사이에 있다면 직진을 하게끔 유도.
                    print("straight")



    cv2.imshow("Source", src) # 원본 이미지
    cv2.imshow("just gray", gray) # 그레이스케일 변환된 이미지
    cv2.imshow("blur_gray", blur_gray) # 블러처리된 이미지
    cv2.imshow("edges", canny_edges) # 케니엣지 적용된 이미지
    cv2.imshow("mask", mask) # 원하는 영역 마스크 이미지
    cv2.imshow("masked", masked_image) # 마스크 영역 내의 엣지 이미지
    cv2.imshow("line", line_img) # 검은 배경에 검출된 라인 그린 이미지

    result = cv2.addWeighted(src, 0.8, line_img, 1, 0) # 원본 이미지에 검출된 라인을 합친 이미지
    cv2.imshow("result", result) # 결과 이미지

    cv2.setMouseCallback("result", onMouse) #마우스 콜백 함수

    cv2.waitKey(5) # 딜레이를 주어 영상 속도 조절

    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q'를 입력하면 무한루프 탈출
        break

cap.release() # 동영상파일 장치 해제
cv2.destroyAllWindows() # 윈도우 파괴