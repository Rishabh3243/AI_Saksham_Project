import { Slider } from "primereact/slider";
import React, { useEffect, useState } from "react";
import Webcam from "react-webcam";
import { Container, Row, Col } from "react-grid-system";
export default function Monitor() {
  const [count, setCount] = useState(0);
  const [messages, setMessages] = useState([]);
  const [emotion, setemotion] = useState({
    div1: false,
    div2: true,
    div3: true,
  });
  const [eyes, setEyes] = useState(20);
  const [imagCode, setImageCode] = useState("");
  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user",
  };
  const [buttonStatus, setButtonStatus] = useState(false);
  const handleClick = (e) => {
    e.preventDefault();
    if (buttonStatus === false) {
      setButtonStatus(true);
    } else {
      setButtonStatus(false);
    }
  };

  const webcamRef = React.useRef(null);
  useEffect(() => {

    if (buttonStatus === true) {
      setTimeout(function(){
      setImageCode(webcamRef.current.getScreenshot());

        const requestOptions = {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code:imagCode})
      };
      fetch('http://127.0.0.1:5001/employees', requestOptions)
          .then(response => response.json())
          .then(data => console.log(data.code));
        setCount((count) => count + 1);
      },1000);
      
    }
  });


  return (
    <div>
      <Col
        style={{ backgroundColor: "green", padding: "10px", width: "cover" }}
      >
        <Row>
          <Col sm={2}>
            <>
              <Row
                style={{
                  paddingLeft: "40px",
                  backgroundColor: "#ADDFFF",
                  fontFamily: "cursive",
                  paddingTop: "3px",
                  paddingBottom: "3px",
                }}
              >
                <b>EMOTION DETECTION</b>
              </Row>
              <Row style={{ backgroundColor: "#ADDFFF" }}>
                <Col sm={4} style={{ marginTop: "5px", marginBottom: "5px" }}>
                  <div
                    style={{
                      backgroundColor: emotion.div1 ? "white" : "",
                      padding: "5px",
                      borderRadius: "5px",
                    }}
                  >
                    😀
                  </div>
                </Col>
                <Col sm={4} style={{ marginTop: "5px", marginBottom: "5px" }}>
                  <div
                    style={{
                      backgroundColor: emotion.div2 ? "white" : "",
                      padding: "5px",
                      borderRadius: "5px",
                    }}
                  >
                    😐
                  </div>
                </Col>
                <Col sm={4} style={{ marginTop: "5px", marginBottom: "5px" }}>
                  <div
                    style={{
                      backgroundColor: emotion.div3 ? "white" : "",
                      padding: "5px",
                      borderRadius: "5px",
                    }}
                  >
                    😡
                  </div>
                </Col>
              </Row>
            </>
            <Row
              style={{
                paddingLeft: "60px",
                backgroundColor: "#B7E9C2",
                fontFamily: "cursive",
                paddingTop: "3px",
                paddingBottom: "3px",
              }}
            >
              <b>EYES OPENNESS</b>
            </Row>
            <Row style={{ backgroundColor: "#B7E9C2" }}>
              <Col sm={2} style={{ marginTop: "5px", marginBottom: "5px" }}>
                0%
              </Col>
              <Col sm={7} style={{ marginTop: "13px", marginBottom: "5px" }}>
                <Slider
                  value={eyes}
                  onChange={(e) => setEyes(e.value)}
                  className="w-14rem"
                />
              </Col>
              <Col sm={3} style={{ marginTop: "5px", marginBottom: "5px" }}>
                100%
              </Col>
            </Row>
            <Row>
              <Col>
                <button onClick={handleClick}>HI</button>
              </Col>
            </Row>
          </Col>
          <Col sm={10} style={{ backgroundColor: "orange" }}>
            <Webcam
              ref={webcamRef}
              width={1180}
              height={720}
              mirrored={true}
              videoConstraints={videoConstraints}
            />
          </Col>
        </Row>
      </Col>
    </div>
  );
}
