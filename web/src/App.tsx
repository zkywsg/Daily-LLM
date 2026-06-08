import { BrowserRouter, Route, Routes } from "react-router";
import { Layout } from "./components/ui/Layout";
import { NotFoundPage } from "./components/ui/NotFoundPage";
import { HomePage } from "./components/home/HomePage";
import { FamilyPage } from "./components/family/FamilyPage";
import { NodePage } from "./components/node/NodePage";

export function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/families/:familyId" element={<FamilyPage />} />
          <Route
            path="/families/:familyId/:nodeSlug"
            element={<NodePage />}
          />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
