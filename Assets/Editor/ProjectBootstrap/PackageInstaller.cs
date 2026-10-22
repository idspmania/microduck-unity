using System;
using System.Linq;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEditor.PackageManager.Requests;
using UnityEngine;

namespace Microduck.Editor
{
    public static class PackageInstaller
    {
        static AddAndRemoveRequest request;
        static double deadline;
        public static void Install()
        {
            request = Client.AddAndRemove(new[] {
                "file:../LocalPackages/org.mujoco", "com.unity.ai.inference@2.6.1"
            });
            deadline = EditorApplication.timeSinceStartup + 600;
            EditorApplication.update += Poll;
        }
        static void Poll()
        {
            if (!request.IsCompleted && EditorApplication.timeSinceStartup < deadline) return;
            EditorApplication.update -= Poll;
            if (request.Status == StatusCode.Success)
            {
                Debug.Log("MICRODUCK_PACKAGES " + string.Join(", ", request.Result.Select(p => p.name + "@" + p.version)));
                EditorApplication.Exit(0);
            }
            else
            {
                Debug.LogError("MICRODUCK_PACKAGES_FAILED " + request.Error?.message);
                EditorApplication.Exit(1);
            }
        }
    }
}
