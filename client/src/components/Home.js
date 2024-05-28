import React from "react";
import { Container, Row, Col } from "react-grid-system";

export default function Home() {
  return (
    <div style={{backgroundImage:"linear-gradient( rgba(147,136,29,1) 0%, rgba(186,174,37,1) 55%, rgba(194,193,68,1) 100%)"}}>
      <Container style={{height:750}} >
        <Row >
        <Col sm={12}>
        <div>ji</div></Col>
          {/*<Col sm={3}>One of three columns</Col>
          <Col sm={3}>One of three columns</Col>
          <Col sm={3}>One of three columns</Col>*/}
        </Row>
      </Container>
    </div>
  );
}
