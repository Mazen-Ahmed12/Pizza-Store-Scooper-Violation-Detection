import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button"; 
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Loader2 } from "lucide-react"; 

function App() {
  const [violationCount, setViolationCount] = useState(0); 
  const [violations, setViolations] = useState<
    { count: number; time: string; frame_id: string; file_path: string }[]
  >([]);
  const [mode, setMode] = useState<"idle" | "processing_video" | "live_rtsp">("idle"); 
  const [liveVideoUrl, setLiveVideoUrl] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const rtspInputRef = useRef<HTMLInputElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const wsUrl = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`; 

  useEffect(() => {
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "reset") {
        setViolationCount(0);
        setViolations([]); 
        return;
      }

      if (data.type === "processing_started") {
        setMode(data.source_type === "rtsp" ? "live_rtsp" : "processing_video");
        setLiveVideoUrl(data.live_url); 
        return;
      }

      if (data.type === "processing_finished") {
        setMode("idle");
        setLiveVideoUrl(null); 
        return;
      }

      // This allows logs to appear IMMEDIATELY during RTSP streaming
      if (data.type === "violation") {
        setViolationCount(data.count || 0);
        setViolations((prev) => [
          {
            count: data.count,
            time: data.time,
            frame_id: data.frame_id,
            file_path: data.file_path,
          },
          ...prev,
        ]);
      }
    };

    return () => wsRef.current?.close(); 
  }, [wsUrl]);

  const uploadAndProcess = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) return alert("Select a video"); 

    const filename = crypto.randomUUID() + "." + file.name.split(".").pop();
    const uploadForm = new FormData();
    uploadForm.append("file", file); 

    const res = await fetch(`/save/${filename}`, { method: "POST", body: uploadForm });
    if (!res.ok) return alert("Upload failed"); 

    const startForm = new FormData();
    startForm.append("path", filename);
    await fetch("/start", { method: "POST", body: startForm }); 
  };

  const startRtsp = async () => {
    const rtsp = rtspInputRef.current?.value.trim();
    if (!rtsp) return alert("Enter RTSP URL"); 

    const form = new FormData();
    form.append("rtsp", rtsp);
    await fetch("/start", { method: "POST", body: form }); 
  };

  const stopProcessing = async () => {
    await fetch("/stop", { method: "POST" }); 
  };

  const isDisabled = mode !== "idle";

  return (
    <div className="min-h-screen bg-gray-100 py-8 px-4">
      {mode === "processing_video" && (
        <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center">
          <div className="bg-white rounded-xl p-8 flex flex-col items-center">
            <Loader2 className="h-16 w-16 animate-spin text-indigo-600 mb-4" />
            <p className="text-xl font-semibold">Processing video...</p> 
          </div>
        </div>
      )}

      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-indigo-700 mb-2">Scooper Violation Detector</h1> 
        <p className="text-2xl mb-8">Violations: <span className="font-bold text-red-600">{violationCount}</span></p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Upload Video</h2>
            <Input ref={fileInputRef} type="file" accept="video/*" disabled={isDisabled} className="mb-4" />
            <Button onClick={uploadAndProcess} disabled={isDisabled} className="w-full mb-8 bg-blue-600">Upload & Process</Button>

            <h2 className="text-xl font-semibold mb-4">RTSP Stream</h2>
            <Input ref={rtspInputRef} type="text" placeholder="rtsp://..." disabled={isDisabled} className="mb-4" />
            
            {isDisabled ? (
              <Button onClick={stopProcessing} className="w-full bg-red-600">Stop Current Job</Button> 
            ) : (
              <Button onClick={startRtsp} className="w-full bg-purple-600">Start RTSP</Button> 
            )}
          </Card>

          <div className="lg:col-span-2 space-y-8">
            {mode === "live_rtsp" && liveVideoUrl && (
              <Card className="p-6">
                <h2 className="text-2xl font-semibold mb-4">Live Stream</h2> 
                <div className="bg-black rounded-lg overflow-hidden aspect-video">
                  <img src={liveVideoUrl} alt="Live Stream" className="w-full h-full object-contain" /> 
                </div>
              </Card>
            )}

            <Card className="p-6">
              <h2 className="text-2xl font-semibold mb-6">Detected Violations</h2> 
              <div className="space-y-6 max-h-[80vh] overflow-y-auto">
                {violations.length === 0 && <p className="text-center text-gray-500 py-10">No violations detected.</p>}
                {violations.map((v, i) => (
                  <div key={i} className="border-l-4 border-red-500 pl-4 bg-white p-4 rounded shadow-sm">
                    <p className="text-sm text-gray-600">{v.time ? new Date(v.time).toLocaleString() : "Just now"} — Violation #{v.count}</p>
                    {v.file_path && (
                      <img src={`/violations/${v.file_path}`} alt="Violation" className="mt-4 max-w-full rounded border" />
                    )}
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;