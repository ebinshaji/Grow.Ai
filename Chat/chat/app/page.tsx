"use client";

import { useChat } from "ai/react";


export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: "/api/llama2",
  });

  

  return (
<div className="flex flex-col w-full max-w-md py-24 mx-auto stretch">
  {messages.map((message) => (
    <div
      key={message.id}
      className="whitespace-pre-wrap"
      style={{
        color: "white",
        backgroundColor: message.role === "user" ? "#57c236" : "#396eff",
        textAlign: message.role === "user" ? "right" : "left",
        alignSelf: message.role === "user" ? "flex-end" : "flex-start",
        maxWidth: "200px",
        padding: "8px",
        borderRadius: "8px",
        margin: "4px" // Optional: Add margin for spacing between messages
      }}
    >
  <strong>{`${message.role}: `}</strong>
  {message.content}
  <br />
  <br />
</div>
      ))}




      <form onSubmit={handleSubmit}>
      <input
      id="DataInput"
  className="fixed bottom-0 p-2 mb-8 border border-white-300 rounded shadow-xl"
  style={{
    backgroundColor: '#09161e',
    color: 'white',
    border: '1px solid white',
    width: '60vw'
  }}
  value={input}
  placeholder="GrowAi- How Can I Help You?"
  onChange={handleInputChange}
/>
      </form>
    </div>
  );
}

