import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

test("shows the 2012 node by default", () => {
  window.location.hash = "";

  render(<App />);

  expect(
    screen.getByRole("heading", { name: /AlexNet：一声炮响/ }),
  ).toBeInTheDocument();
});

test("clicking a timeline node updates the content and URL hash", async () => {
  window.location.hash = "";

  render(<App />);

  await userEvent.click(
    screen.getByRole("button", { name: /2017 Transformer/ }),
  );

  expect(
    screen.getByRole("heading", { name: /Transformer：把 RNN 扔掉/ }),
  ).toBeInTheDocument();
  expect(window.location.hash).toBe("#2017");
});
